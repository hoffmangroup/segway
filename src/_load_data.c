/* XXXopt: probably the most important place for optimization is chunk
   space, but you should do profiling to see whether writing is the
   slowest bit

   another optimization would be to inflate data yourself, then try to
   process multiple input tracks in parallel (reduce number of writes
   by num_cols)

   uncompressing and recompressing all the data for every new track is
   probably pretty expensive
*/

#define _GNU_SOURCE

#include <argp.h>
#include <assert.h>
#include <error.h>
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

#define NARGS 2
#define CARDINALITY 2

/* XXX: this needs to adjust, but always be smaller than max size for a dataset*/
#define CHUNK_NROWS 10000
#define DEFLATE_LEVEL 1

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

void set_attr(hid_t loc, const char *name, hid_t datatype, hid_t dataspace,
              const void *buf) {
  hid_t attr;

  attr = H5Acreate(loc, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
  assert(attr >= 0);

  assert(H5Awrite(attr, datatype, buf) >= 0);

  assert(H5Aclose(attr) >= 0);
}

/* set an attr from a C string */
void set_attr_str(hid_t loc, const char *name, const char *buf) {
  hid_t datatype, dataspace, mem_type_id;
  hsize_t buf_len;

  /* create datatype */
  datatype = H5Tcopy(H5T_C_S1);
  assert(datatype >= 0);

  buf_len = strlen(buf);
  /* 0 length in datatype size not supported */
  if (buf_len == 0) {
    buf_len = 1;
  }
  assert(H5Tset_size(datatype, buf_len) >= 0);

  /* create dataspace */
  dataspace = H5Screate(H5S_SCALAR);
  assert(dataspace >= 0);

  mem_type_id = H5T_STRING;

  set_attr(loc, name, datatype, dataspace, buf);
}

/* fetch num_cols and the col for a particular trackname */
void get_cols(hid_t h5file, char *trackname, hsize_t *num_cols,
              hsize_t *col) {
  hid_t attr, root, dataspace, datatype;
  hsize_t data_size, cell_size, num_cells;
  char *attr_data;

  root = H5Gopen(h5file, "/", H5P_DEFAULT);
  assert(root >= 0);

  attr = H5Aopen_name(root, "tracknames");
  assert(attr >= 0);

  dataspace = H5Aget_space(attr);
  assert(dataspace >= 0);

  assert(H5Sget_simple_extent_dims(dataspace, num_cols, NULL) == 1);
  assert(H5Sclose(dataspace) >= 0);

  datatype = H5Aget_type(attr);
  assert(datatype >= 0);
  assert(H5Tget_class(datatype) == H5T_STRING);

  cell_size = H5Tget_size(datatype);
  assert(cell_size >= 0);

  data_size = H5Aget_storage_size(attr);
  assert(data_size >= 0);

  num_cells = data_size / cell_size;

  attr_data = alloca(data_size);
  assert(attr_data);

  assert(H5Aread(attr, datatype, attr_data) >= 0);

  *col = 0;
  for (*col = 0; *col <= num_cells; (*col)++) {
    if (*col == num_cells) {
      fprintf(stderr, "can't find trackname: %s\n", trackname);
      exit(EXIT_FAILURE);
    } else {
      if (!strncmp(attr_data + (*col * cell_size), trackname, cell_size)) {
        break;
      }
    }
  }

  assert(H5Aclose(attr) >= 0);
  assert(H5Gclose(root) >= 0);
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

  get_attr(subgroup, ATTR_START, H5T_STD_I32LE, &supercontig->start);
  get_attr(subgroup, ATTR_END, H5T_STD_I32LE, &supercontig->end);
  supercontig->group = subgroup;

  fprintf(stderr, " %s (%d, %d): %d\n", name, supercontig->start,
          supercontig->end, subgroup);

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
  assert(save_ptr);

  newstring = line + strlen(FMT_WIGFIX);

  while ((token = strtok_r(newstring, DELIM_WIG, &save_ptr))) {
    loc_eq = strchr(token, '=');
    key = strndupa(token, loc_eq - token);
    assert(key);

    val = loc_eq + 1;

    if (!strcmp(key, KEY_CHROM)) {
      *chrom = strdup(val);
      assert(*chrom);
    } else if (!strcmp(key, KEY_START)) {
      *start = strtol(val, &tailptr, 10) - 1;
      assert(!*tailptr);
    } else if (!strcmp(key, KEY_STEP)) {
      *step = strtol(val, &tailptr, 10);
      assert(!*tailptr);
    } else {
      fprintf(stderr, "can't understand key: %s", key);
      exit(1);
    }

    newstring = NULL;
  }
}

void init_supercontig_array(size_t len, supercontig_array_t *supercontigs) {
  supercontigs->len = len;

  supercontigs->supercontigs = malloc(len * sizeof(supercontig_t));
  assert(supercontigs->supercontigs);

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
  assert(H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data) >= 0);
  assert(H5Eset_auto(H5E_DEFAULT, NULL, NULL) >= 0);

  dataset = H5Dopen(loc, name, dapl);

  /* re-enable errors */
  assert(H5Eset_auto(H5E_DEFAULT, old_func, old_client_data) >= 0);

  return dataset;
}

/* make existing dataset into a PyTables CArray by setting appropriate
   attrs */
void make_pytables_carray(hid_t dataset) {
  set_attr_str(dataset, "CLASS", "CARRAY");
  set_attr_str(dataset, "TITLE", "");
  set_attr_str(dataset, "VERSION", "1.0");
}

void write_buf(hid_t h5file, char *trackname, float *buf_start, float *buf_end,
               float *buf_filled_start, float *buf_filled_end,
               supercontig_array_t *supercontigs) {
  size_t buf_offset_start, buf_offset_end;
  hid_t dataset = -1;

  hid_t mem_dataspace = -1;
  hid_t file_dataspace = -1;

  hsize_t num_cols, col;

  hsize_t mem_dataspace_dims[CARDINALITY] = {-1, 1};
  hsize_t file_dataspace_dims[CARDINALITY];
  hsize_t select_start[CARDINALITY];
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
      buf_offset_end = supercontig->end;
    }
    if (buf_offset_end < supercontig->start) {
      continue;
    }
    assert (buf_offset_end <= supercontig->end);

    /* set mem dataspace */
    mem_dataspace_dims[0] = buf_offset_end - buf_offset_start;
    mem_dataspace = H5Screate_simple(1, mem_dataspace_dims, NULL);
    assert(mem_dataspace >= 0);

    /* calc dimensions */
    get_cols(h5file, trackname, &num_cols, &col);

    /* set dataset if it exists */
    dataset = open_dataset(supercontig->group, DATASET_NAME, H5P_DEFAULT);

    if (dataset >= 0) {
      /* set file dataspace */
      file_dataspace = H5Dget_space(dataset);
      assert(file_dataspace >= 0);
    } else {
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
      assert(H5Pset_deflate(dataset_creation_plist, DEFLATE_LEVEL) >= 0);

      /* create dataset */
      fprintf(stderr, "creating %lld x %lld dataset in %d...",
             file_dataspace_dims[0], file_dataspace_dims[1],
             supercontig->group);
      dataset = H5Dcreate(supercontig->group, DATASET_NAME, DTYPE,
                          file_dataspace, H5P_DEFAULT,
                          dataset_creation_plist, H5P_DEFAULT);
      assert(dataset >= 0);
      fprintf(stderr, " done\n");

      make_pytables_carray(dataset);
    }

    /* XXX: set dirty attribute */
    /* XXX: need to look up boolean attributes, etc. */

    /* select file hyperslab */
    select_start[0] = buf_offset_start - supercontig->start;
    select_start[1] = col;

    /* count has same dims as mem_dataspace */
    assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                               NULL, mem_dataspace_dims, NULL) >= 0);

    /* write */
    fprintf(stderr, "writing %lld floats...", mem_dataspace_dims[0]);
    assert(H5Dwrite(dataset, DTYPE, mem_dataspace, file_dataspace,
                    H5P_DEFAULT, buf_filled_start) >= 0);
    fprintf(stderr, " done\n");

    assert(H5Dclose(dataset) >= 0);

    /* close dataspaces */
    close_dataspace(file_dataspace);
    close_dataspace(mem_dataspace);
  }

  assert(H5Pclose(dataset_creation_plist) >= 0);
}

void proc_wigfix_header(char *line, char *h5dirname, hid_t *h5file,
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

  fprintf(stderr, "%s (%ld)\n", chrom, start);

  /* set h5filename */
  h5filename = alloca(strlen(h5dirname)+strlen(chrom)+strlen(SUFFIX_H5)+2);
  assert(h5filename);

  suffix = stpcpy(h5filename, h5dirname);
  suffix = stpcpy(suffix, "/");
  suffix = stpcpy(suffix, chrom);
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
  assert(*buf_start);

  *buf_ptr = *buf_start + start;
  *buf_end = *buf_start + buf_len;
}

void load_data(char *h5dirname, char *trackname) {
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

  if (getline(&line, &size_line, stdin) < 0) {
    error_at_line(EXIT_FAILURE, errno, __FILE__, __LINE__,
                  "failed to read first line");
  }

  proc_wigfix_header(line, h5dirname, &h5file, &supercontigs,
                     &buf_start, &buf_end, &buf_ptr);
  buf_filled_start = buf_ptr;

  while (getline(&line, &size_line, stdin) >= 0) {
    datum = strtof(line, &tailptr);
    if (*tailptr == '\n') {
      if (buf_ptr < buf_end) {
        *buf_ptr++ = datum;
      } /* else: ignore data until we get to another header line */
    } else {
      write_buf(h5file, trackname, buf_start, buf_end, buf_filled_start,
                buf_ptr, &supercontigs);
      proc_wigfix_header(line, h5dirname, &h5file, &supercontigs,
                         &buf_start, &buf_end, &buf_ptr);
      buf_filled_start = buf_ptr;
    }
  }


  write_buf(h5file, trackname, buf_start, buf_end, buf_filled_start, buf_ptr,
            &supercontigs);

  /* free heap variables */
  free_supercontig_array(&supercontigs);
  free(line);
  free(buf_start);

  close_file(h5file);
}

const char *argp_program_version = "$Revision$";
const char *argp_program_bug_address = "Michael Hoffman <mmh1@washington.edu>";

static char doc[] = "A fast loader of genomic data into HDF5";
static char args_doc[] = "DST TRACKNAME";

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
  char **arguments = state->input;

  switch (key) {
  case ARGP_KEY_ARG:
    if (state->arg_num >= NARGS) {
      argp_usage(state);
      exit(EXIT_FAILURE);
    }
    arguments[state->arg_num] = arg;
    break;

  case ARGP_KEY_END:
    if (state->arg_num < NARGS) {
      argp_usage(state);
      exit(EXIT_FAILURE);
    }

    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}


static struct argp argp = {0, parse_opt, args_doc, doc};

int main(int argc, char **argv) {
  char *arguments[NARGS];
  char *h5dirname, *trackname;

  assert(argp_parse(&argp, argc, argv, 0, 0, arguments) == 0);

  h5dirname = arguments[0];
  trackname = arguments[1];

  load_data(h5dirname, trackname);

  return EXIT_SUCCESS;
}

