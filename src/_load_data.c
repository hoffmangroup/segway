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


typedef struct {
  size_t len;
  supercontig_t *supercontigs;
  supercontig_t *supercontig_curr;
} supercontig_array_t;

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
  supercontig_t *supercontig;
  supercontig_array_t *supercontigs;

  supercontigs = (supercontig_array_t *) op_info;

  supercontig = supercontigs->supercontig_curr++;

  /* leave open */
  printf("%s\n", name);
  subgroup = H5Gopen(g_id, name, H5P_DEFAULT);
  assert(subgroup >= 0);

  get_attr(subgroup, ATTR_START, H5T_STD_I32LE, &supercontig->start);
  get_attr(subgroup, ATTR_END, H5T_STD_I32LE, &supercontig->end);
  supercontig->group = subgroup;

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

void close_file(hid_t h5file) {
  if (h5file >= 0) {
    assert(H5Fclose(h5file) >= 0);
  }
}

void parse_wigfix_header(char *line, char **chrom, long *start, long *step) {
  /* mallocs chrom; caller must free() it */

  char *save_ptr;
  char *token;
  char *tailptr;
  char *newstring;

  char *loc_eq;
  char *key;
  char *val;

  assert(!strncmp(FMT_WIGFIX, line, LEN_FMT_WIGFIX));

  save_ptr = strdupa(line);
  newstring = line + LEN_FMT_WIGFIX;

  while (token = (strtok_r(newstring, DELIM_WIG, &save_ptr))) {
    loc_eq = strchr(token, '=');
    key = strndup(token, loc_eq - token);
    val = loc_eq + 1;

    if (!strcmp(key, KEY_CHROM)) {
      *chrom = strdup(val);
    } else if (!strcmp(key, KEY_START)) {
      *start = strtol(val, &tailptr, 10);
      assert(!*tailptr);
    } else if (!strcmp(key, KEY_STEP)) {
      *step = strtol(val, &tailptr, 10);
    } else {
      printf("can't understand key: %s", key);
      exit(1);
    }

    free(key);

    newstring = NULL;
  }
}

void init_supercontig_array(size_t len,
                            supercontig_array_t *supercontigs) {
  supercontigs->len = len;
  supercontigs->supercontigs = malloc(len * sizeof(supercontig_t));
  supercontigs->supercontig_curr = supercontigs->supercontigs;
}

#if 0
void write_XXX(XXX) {
  /* for error suppression */
  H5E_auto2_t old_func;
  void *old_client_data;

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

    chunk_dims[1] = num_cols;
    assert(H5Pset_chunk(dataset_creation_plist, CARDINALITY, chunk_dims)
           >= 0);

    printf("creating %lld x %lld dataset\n", dims[0], dims[1]);
    dataset = H5Dcreate(supercontig.group, DATASET_NAME, DTYPE,
                        file_dataspace, H5P_DEFAULT,
                        dataset_creation_plist, H5P_DEFAULT);
    assert(dataset >= 0);
    printf("done\n");
  }

      /*
     at end, write to supercontigs in segments

      XXX: select new slab and write
      assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                                 NULL, select_count, NULL) >= 0);

      assert(H5Dwrite(dataset, DTYPE, mem_dataspace, file_dataspace,
                      H5P_DEFAULT, &datum) >= 0);
      */

}
#endif

void proc_wigfix_header(char *line, hid_t *h5file,
                        supercontig_array_t *supercontigs,
                        float **buf, size_t *buf_len) {
  long start = -1;
  long step = 1;

  char *chrom = NULL;
  char *h5filename = NULL;

  hid_t root = -1;

  H5G_info_t root_info;

  /* do writing if buf_len > 0 */

  parse_wigfix_header(line, &chrom, &start, &step);
  assert(chrom && start >= 0 && step == 1);

  printf("%s (%ld)\n", chrom, start);

  /* set h5filename */
  h5filename = strndupa(chrom, strlen(chrom)+strlen(EXT_H5));
  strcpy(h5filename+strlen(chrom), EXT_H5);
  free(chrom);

  /* XXXopt: don't close if it's the same file */
  close_file(*h5file);

  /* open the chromosome file */
  *h5file = H5Fopen(h5filename, H5F_ACC_RDWR, H5P_DEFAULT);
  assert(*h5file >= 0);

  root = H5Gopen(*h5file, "/", H5P_DEFAULT);
  assert(root >= 0);

  /* allocate supercontig metadata array */
  assert(H5Gget_info(root, &root_info) >= 0);
  init_supercontig_array(root_info.nlinks, supercontigs);

  /* populate supercontig metadata array */
  assert(H5Literate(root, H5_INDEX_NAME, H5_ITER_INC, NULL,
                    supercontig_visitor, &supercontigs) == 0);

  assert(H5Gclose(root) >= 0);

  /* allocate buffer: enough to assign values from 0 to the end of the
     last supercontig */
  /* XXX: need to ensure sorting */
  *buf_len = ((supercontigs->supercontigs)[supercontigs->len-1]).end;
  *buf = malloc(*buf_len * sizeof(float));
}

int main(void) {
  char *line;
  size_t size_line = 0;
  char *tailptr;

  size_t buf_len = 0;
  float *buf, *buf_ptr, *buf_end;

  supercontig_array_t supercontigs;

  float datum;

  hid_t h5file = -1;
  hid_t dataset = -1;
  hid_t dataset_creation_plist = -1;
  hid_t mem_dataspace = -1;
  hid_t file_dataspace = -1;

  hsize_t num_cols;

  hsize_t dims[CARDINALITY];
  hsize_t select_start[CARDINALITY] = {-1, COL};
  hsize_t select_count[CARDINALITY] = {1, 1};
  hsize_t chunk_dims[CARDINALITY] = {1000000, -1};

  dataset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);
  assert(dataset_creation_plist >= 0);

  assert(H5Pset_fill_value(dataset_creation_plist, DTYPE, &nan_float) >= 0);

  mem_dataspace = H5Screate(H5S_SCALAR);
  assert(mem_dataspace >= 0);

  /* XXXopt: would be faster to just read a big block and do repeated
     strtof rather than using getline */

  assert (getline(&line, &size_line, stdin) >= 0);
  proc_wigfix_header(line, &h5file, &supercontigs, &buf, &buf_len);

  while (getline(&line, &size_line, stdin) >= 0) {
    datum = strtof(line, &tailptr);
    if (*tailptr == '\n') {
      if (buf_ptr < buf_end) {
        *buf_ptr++ = datum;
      } /* else: ignore the data */
    } else {
      /* strip trailing newline */
      *strchr(line, '\n') = '\0';

      proc_wigfix_header(line, &h5file, &supercontigs, &buf, &buf_len);

      buf_ptr = buf;
      buf_end = buf_ptr + buf_len;
    }
  }

  close_dataspace(mem_dataspace);
  close_file(h5file);

  assert(H5Pclose(dataset_creation_plist) >= 0);

  return 0;
}

