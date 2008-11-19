#define _GNU_SOURCE

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

#define FMT_WIGFIX "fixedStep "
#define LEN_FMT_WIGFIX strlen(FMT_WIGFIX)
#define DELIM_WIG " "

#define KEY_CHROM "chrom"
#define KEY_START "start"
#define KEY_STEP "step"

#define ATTR_START "start"
#define ATTR_END "end"

#define DATASET_NAME "continuous"
#define DTYPE H5T_IEEE_F32LE

#define EXT_H5 ".h5"

#define CARDINALITY 2
#define COL 0 /* XXX: this really can't be hard-coded, just for debugging */

const float nan_float = NAN;

typedef struct {
  int start;
  int end;
  hid_t group;
} supercontig_t;

void get_attr(hid_t loc, const char *name, hid_t mem_type_id, void *buf) {
  hid_t attr;

  attr = H5Aopen_name(loc, name);
  assert(attr >= 0);

  assert(H5Aread(attr, mem_type_id, buf) >= 0);

  assert(H5Aclose(attr) >= 0);
}

hsize_t get_num_cols(hid_t h5file) {
  hid_t attr, root, dataspace;
  hsize_t dim;

  root = H5Gopen(h5file, "/", H5P_DEFAULT);
  assert(root >= 0);

  attr = H5Aopen_name(root, "tracknames");
  assert(attr >= 0);

  dataspace = H5Aget_space(attr);
  assert(dataspace >= 0);

  assert (H5Sget_simple_extent_dims(dataspace, &dim, NULL) == 1);

  assert(H5Sclose(dataspace) >= 0);
  assert(H5Aclose(attr) >= 0);
  assert(H5Gclose(root) >= 0);

  return dim;
}

herr_t supercontig_visitor(hid_t g_id, const char *name,
                           const H5L_info_t *info,
                           void *op_info) {
  hid_t subgroup;
  int start;
  int end;
  supercontig_t *supercontig;

  supercontig = (supercontig_t *) op_info;

  subgroup = H5Gopen(g_id, name, H5P_DEFAULT);
  assert(subgroup >= 0);

  get_attr(subgroup, ATTR_START, H5T_STD_I32LE, &start);
  get_attr(subgroup, ATTR_END, H5T_STD_I32LE, &end);

  if (start <= supercontig->start && supercontig->start < end) {
    supercontig->start = start;
    supercontig->end = end;
    supercontig->group = subgroup;
    return 1;
  }

  assert(H5Gclose(subgroup) >= 0);
  return 0;
}

void close_dataset(hid_t dataset) {
  if (dataset >= 0) {
    assert(H5Dclose(dataset) >= 0);
  }
}

void close_dataspace(hid_t dataspace) {
  if (dataspace >= 0) {
    assert(H5Sclose(dataspace) >= 0);
  }
}

void close_group(hid_t group) {
  if (group >= 0) {
    assert(H5Gclose(group) >= 0);
  }
}

int main(void) {
  char *save_ptr;
  char *line;
  char *token;
  char *newstring;
  size_t size_line = 0;
  char *tailptr;

  char *loc_eq;
  char *key;
  char *val;
  size_t len_key;

  char *chrom;
  char *h5filename = NULL;
  long new_start, offset;

  supercontig_t supercontig;

  float datum;

  hid_t h5file = -1;
  hid_t dataset = -1;
  hid_t dataset_creation_plist = -1;
  hid_t mem_dataspace = -1;
  hid_t file_dataspace = -1;

  supercontig.group = -1;

  hsize_t num_cols;

  hsize_t dims[CARDINALITY];
  hsize_t select_start[CARDINALITY] = {-1, COL};
  hsize_t select_count[CARDINALITY] = {1, 1};

  /* for error suppression */
  H5E_auto2_t old_func;
  void *old_client_data;

  dataset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);
  assert(dataset_creation_plist >= 0);

  assert(H5Pset_fill_value(dataset_creation_plist, DTYPE, &nan_float) >= 0);

  mem_dataspace = H5Screate(H5S_SCALAR);
  assert(mem_dataspace >= 0);

  while (getline(&line, &size_line, stdin) >= 0) {
    *strchr(line, '\n') = '\0';

    datum = strtof(line, &tailptr);
    if (!*tailptr && h5file >= 0) {
      select_start[0] = offset;
      assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                                 NULL, select_count, NULL) >= 0);

      /* XXXopt: write in memory instead of each time */
      printf("writing %f...", datum);
      assert(H5Dwrite(dataset, DTYPE, mem_dataspace, file_dataspace,
                      H5P_DEFAULT, &datum) >= 0);
      printf(" done\n");
      offset++;
    } else {
      /* XXX: most of this should go to another function:
         parse_wigfix_header() */
      assert(!strncmp(FMT_WIGFIX, line, LEN_FMT_WIGFIX));

      h5filename = NULL;
      new_start = -1;

      save_ptr = strdupa(line);
      newstring = line+LEN_FMT_WIGFIX;

      while (token = strtok_r(newstring, DELIM_WIG, &save_ptr)) {
        loc_eq = strchr(token, '=');
        key = strndup(token, loc_eq - token);
        val = loc_eq + 1;

        if (!strcmp(key, KEY_CHROM)) {
          chrom = strdupa(val);
          h5filename = strndupa(val, strlen(val)+strlen(EXT_H5));

          strcpy(h5filename+strlen(val), EXT_H5);

        } else if (!strcmp(key, KEY_START)) {
          new_start = strtol(val, &tailptr, 10);
          assert(!*tailptr);
        } else if (!strcmp(key, KEY_STEP)) {
          assert(!strcmp(val, "1"));
        } else {
          printf("can't understand key: %s", key);
          exit(1);
        }

        newstring = NULL;
      }

      assert(h5filename && new_start >= 0);

      /* XXXopt: don't close if it's the same file or supercontig */
      /* XXX: write dataset in memory */
      close_dataspace(file_dataspace);
      close_dataset(dataset);
      close_group(supercontig.group);
      close_group(h5file);

      printf("%s (%ld)\n", chrom, new_start);

      h5file = H5Fopen(h5filename, H5F_ACC_RDWR, H5P_DEFAULT);
      assert(h5file >= 0);

      supercontig.start = new_start;
      if (H5Lvisit(h5file, H5_INDEX_NAME, H5_ITER_INC, supercontig_visitor,
                   &supercontig) >= 1) {
        printf(" (%d, %d)\n", supercontig.start, supercontig.end);
      }

      /* suppress errors */
      H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
      H5Eset_auto(H5E_DEFAULT, NULL, NULL);

      dataset = H5Dopen(supercontig.group, DATASET_NAME, H5P_DEFAULT);

      /* re-enable errors */
      H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

      if (dataset >= 0) {
        file_dataspace = H5Dget_space(dataset);
        assert(file_dataspace >= 0);
      } else {
        file_dataspace = H5Screate(H5S_SIMPLE);
        assert(file_dataspace >= 0);

        num_cols = get_num_cols(h5file);
        printf("num cols: %lld\n", num_cols);

        dims[0] = supercontig.end - supercontig.start;
        dims[1] = num_cols;

        assert(H5Sset_extent_simple(file_dataspace, CARDINALITY, dims, dims)
               >= 0);

        printf("creating %lld x %lld dataset\n", dims[0], dims[1]);

        dataset = H5Dcreate(supercontig.group, DATASET_NAME, DTYPE,
                            file_dataspace, H5P_DEFAULT,
                            dataset_creation_plist, H5P_DEFAULT);
        assert(dataset >= 0);
        printf("done\n");

        offset = new_start - supercontig.start;
      }
    }
  }

  /* XXXwrite dataset in memory */
  close_dataspace(mem_dataspace);
  close_dataspace(file_dataspace);
  close_dataset(dataset);
  close_group(supercontig.group);
  close_group(h5file);

  assert(H5Pclose(dataset_creation_plist) >= 0);
}

