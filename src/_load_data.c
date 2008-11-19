#define _GNU_SOURCE

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

#define FMT_WIGFIX "fixedStep "
#define DELIM_WIG " "

#define KEY_CHROM "chrom"
#define KEY_START "start"
#define KEY_STEP "step"

#define ATTR_START "start"
#define ATTR_END "end"

#define DATASET_NAME "continuous"
#define DTYPE H5T_IEEE_F32LE

#define SUFFIX_H5 ".h5"

#define CARDINALITY 2
#define COL 0 /* XXX: this really can't be hard-coded, just for debugging */
#define CHUNK_NROWS 1000000

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
  subgroup = H5Gopen(g_id, name, H5P_DEFAULT);
  assert(subgroup >= 0);

  printf("%s: %d\n", name, group);

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

  /* strip trailing newline */
  *strchr(line, '\n') = '\0';

  assert(!strncmp(FMT_WIGFIX, line, strlen(FMT_WIGFIX)));

  save_ptr = strdupa(line);
  newstring = line + strlen(FMT_WIGFIX);

  while ((token = strtok_r(newstring, DELIM_WIG, &save_ptr))) {
    loc_eq = strchr(token, '=');
    key = strndupa(token, loc_eq - token);
    val = loc_eq + 1;

    if (!strcmp(key, KEY_CHROM)) {
      *chrom = strdup(val);
    } else if (!strcmp(key, KEY_START)) {
      *start = strtol(val, &tailptr, 10);
      assert(!*tailptr);
    } else if (!strcmp(key, KEY_STEP)) {
      *step = strtol(val, &tailptr, 10);
      assert(!*tailptr);
    } else {
      printf("can't understand key: %s", key);
      exit(1);
    }

    newstring = NULL;
  }
}

void init_supercontig_array(size_t len,
                            supercontig_array_t *supercontigs) {
  supercontigs->len = len;
  supercontigs->supercontigs = malloc(len * sizeof(supercontig_t));
  supercontigs->supercontig_curr = supercontigs->supercontigs;
}

void free_supercontig_array(supercontig_array_t *supercontigs) {
  for (supercontig_t *supercontig = supercontigs->supercontigs;
       supercontig < supercontigs->supercontigs + supercontigs-> len;
       supercontig++) {
    assert(H5Gclose(supercontig->group) >= 0);
  }
  free(supercontigs->supercontigs);
}

/* suppresses errors */
hid_t open_dataset(hid_t loc, char *name, hid_t dapl) {
  hid_t dataset;

  /* for error suppression */
  H5E_auto2_t old_func;
  void *old_client_data;

  /* suppress errors */
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  dataset = H5Dopen(loc, name, dapl);

  /* re-enable errors */
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

  return dataset;
}

void write_buf(hid_t h5file, float *buf_start, float *buf_end,
               float *buf_filled_start, float *buf_filled_end,
               supercontig_array_t *supercontigs) {
  float *buf_filled_end_ptr; /* for one supercontig */
  size_t buf_offset_start, buf_offset_end;
  hid_t dataset = -1;

  hid_t mem_dataspace = -1;
  hid_t file_dataspace = -1;

  hsize_t num_cols;

  hsize_t mem_dataspace_dims[1];
  hsize_t file_dataspace_dims[CARDINALITY];
  hsize_t select_start[CARDINALITY] = {-1, COL};
  hsize_t select_count[CARDINALITY] = {1, 1};
  hsize_t chunk_dims[CARDINALITY] = {CHUNK_NROWS, -1};

  hid_t dataset_creation_plist = -1;

  dataset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);
  assert(dataset_creation_plist >= 0);

  assert(H5Pset_fill_value(dataset_creation_plist, DTYPE, &nan_float) >= 0);

  for (supercontig_t *supercontig = supercontigs->supercontigs;
       supercontig < supercontigs->supercontigs + supercontigs-> len;
       supercontig++) {

    /* find the start that fits into this supercontig */
    buf_offset_start = buf_filled_start - buf_start;
    if (buf_offset_start < supercontig->start) {
      buf_filled_start = buf_start + supercontig->start;
      buf_offset_start = supercontig->start;
    }
    if (buf_offset_start >= supercontig->end) {
      continue;
    }
    assert (buf_offset_start >= supercontig->start);

    /* find the end that fits into this supercontig */
    buf_offset_end = buf_filled_end - buf_start;
    if (buf_offset_end > supercontig->end) {
      buf_filled_end_ptr = buf_start + supercontig->end;
      buf_offset_end = supercontig->end;
    }
    if (buf_offset_end < supercontig->start) {
      continue;
    }
    assert (buf_offset_end <= supercontig->end);

    /* XXX: set mem dataspace */
    mem_dataspace_dims[0] = buf_offset_end - buf_offset_start;
    mem_dataspace = H5Screate_simple(1, mem_dataspace_dims, NULL);
    assert(mem_dataspace >= 0);

    dataset = open_dataset(supercontig->group, DATASET_NAME, H5P_DEFAULT);

    if (dataset >= 0) {
      /* set file dataspace */
      file_dataspace = H5Dget_space(dataset);
      assert(file_dataspace >= 0);
    } else {
      /* calc dimensions */
      num_cols = get_num_cols(h5file);

      file_dataspace_dims[0] = supercontig->end - supercontig->start;
      file_dataspace_dims[1] = num_cols;

      /* create dataspace */
      file_dataspace = H5Screate_simple(CARDINALITY, file_dataspace_dims,
                                        NULL);
      assert(file_dataspace >= 0);

      /* create chunkspace */
      chunk_dims[1] = num_cols;
      assert(H5Pset_chunk(dataset_creation_plist, CARDINALITY, chunk_dims)
             >= 0);

      /* create dataset */
      printf("creating %lld x %lld dataset in %d\n",
             file_dataspace_dims[0], file_dataspace_dims[1],
             supercontig->group);
      dataset = H5Dcreate(supercontig->group, DATASET_NAME, DTYPE,
                          file_dataspace, H5P_DEFAULT,
                          dataset_creation_plist, H5P_DEFAULT);
      assert(dataset >= 0);
      printf("done\n");

      /* XXX: set PyTables attrs: new func */
    }

    /*
      at end, write to supercontigs in segments

      XXX: select new slab and write
      assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                                 NULL, select_count, NULL) >= 0);

      assert(H5Dwrite(dataset, DTYPE, mem_dataspace, file_dataspace,
                      H5P_DEFAULT, &datum) >= 0);
    */

    close_dataspace(file_dataspace);
    close_dataspace(mem_dataspace);
  }

  assert(H5Pclose(dataset_creation_plist) >= 0);
}

void proc_wigfix_header(char *line, hid_t *h5file,
                        supercontig_array_t *supercontigs,
                        float **buf_start, float **buf_end, float **buf_ptr) {
  long start = -1;
  long step = 1;
  size_t buf_len;

  char *chrom = NULL;
  char *h5filename = NULL;
  char *suffix;

  hid_t root = -1;
  hsize_t idx = 0;

  H5G_info_t root_info;

  /* do writing if buf_len > 0 */
  parse_wigfix_header(line, &chrom, &start, &step);

  assert(chrom && start >= 0 && step == 1);

  printf("%s (%ld)\n", chrom, start);

  /* set h5filename */
  h5filename = alloca(strlen(chrom)+strlen(SUFFIX_H5)+1);
  suffix = stpcpy(h5filename, chrom);
  strcpy(suffix, SUFFIX_H5);
  free(chrom);

  /* XXXopt: don't close if it's the same file */
  if (*buf_start) {
    free(*buf_start);
    free_supercontig_array(supercontigs);
  }
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
  assert(H5Literate(root, H5_INDEX_NAME, H5_ITER_INC, &idx,
                    supercontig_visitor, supercontigs) == 0);

  assert(H5Gclose(root) >= 0);

  /* allocate buffer: enough to assign values from 0 to the end of the
     last supercontig */
  /* XXX: need to ensure sorting */
  buf_len = ((supercontigs->supercontigs)[supercontigs->len-1]).end;

  *buf_start = malloc(buf_len * sizeof(float));
  *buf_ptr = *buf_start + start;
  *buf_end = *buf_start + buf_len;
}

int main(void) {
  char *line = NULL;
  size_t size_line = 0;
  char *tailptr;

  float *buf_start = NULL;
  float *buf_filled_start, *buf_ptr, *buf_end;

  supercontig_array_t supercontigs;

  float datum;

  hid_t h5file = -1;

  /* XXXopt: would be faster to just read a big block and do repeated
     strtof rather than using getline */

  assert (getline(&line, &size_line, stdin) >= 0);

  proc_wigfix_header(line, &h5file, &supercontigs,
                     &buf_start, &buf_end, &buf_ptr);
  buf_filled_start = buf_ptr;

  while (getline(&line, &size_line, stdin) >= 0) {
    datum = strtof(line, &tailptr);
    if (*tailptr == '\n') {
      if (buf_ptr < buf_end) {
        *buf_ptr++ = datum;
      } /* else: ignore data until we get to another header line */
    } else {
      write_buf(h5file, buf_start, buf_end, buf_filled_start, buf_ptr, &supercontigs);
      proc_wigfix_header(line, &h5file, &supercontigs,
                         &buf_start, &buf_end, &buf_ptr);
      buf_filled_start = buf_ptr;
    }
  }

  /* XXX: write_buf() */
  free_supercontig_array(&supercontigs);
  free(line);
  free(buf_start);

  close_file(h5file);

  return 0;
}

