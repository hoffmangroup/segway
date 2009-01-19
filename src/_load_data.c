/* XXXopt: probably the most important place for optimization is chunk
   space, but you should do profiling to see whether writing is the
   slowest bit

   another optimization would be to inflate data yourself, then try to
   process multiple input tracks in parallel (reduce number of writes
   by num_cols)

   doing this plus parallel HDF5 would probably get an enormous speedup
*/

/** includes **/

#define _GNU_SOURCE

#include <argp.h>
#include <assert.h>
#include <error.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

/** constants **/

#define ID_WIGVAR "variableStep "
#define ID_WIGFIX "fixedStep "
#define DELIM_WIG " "

#define KEY_CHROM "chrom"
#define KEY_START "start"
#define KEY_STEP "step"
#define KEY_SPAN "span"

#define ATTR_START "start"
#define ATTR_END "end"

#define DATASET_NAME "continuous"
#define DTYPE H5T_IEEE_F32LE

#define SUFFIX_H5 ".h5"

#define NARGS 2
#define CARDINALITY 2
#define BASE 10

/* XXX: this needs to adjust, but always be smaller than max size for
   a dataset */
#define CHUNK_NROWS 10000

const float nan_float = NAN;

/** typedefs **/

typedef enum {
  FMT_BED, FMT_WIGFIX, FMT_WIGVAR
} file_format;

typedef struct {
  int start;
  int end;
  hid_t group;
} supercontig_t;

typedef struct {
  hid_t h5file; /* handle to the file */
  char *chrom; /* name of chromosome */
  size_t num_supercontigs;
  supercontig_t *supercontigs;
  supercontig_t *supercontig_curr;
} chromosome_t;

/** helper functions **/

/* XXX: GNU-only extension; should be wrapped */
__attribute__((noreturn)) void fatal(char *msg) {
  fprintf(stderr, msg);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

void *xmalloc(size_t size)
{
  register void *value = malloc(size);
  if (value == 0)
    fatal("virtual memory exhausted");
  return value;
}

/** general-purpose HDF5 attribute helper functions **/

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

/** dataset **/

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

void close_dataset(hid_t dataset) {
  if (dataset >= 0) {
    assert(H5Dclose(dataset) >= 0);
  }
}

/** dataspace **/

hid_t get_file_dataspace(hid_t dataset) {
  hid_t dataspace;

  dataspace = H5Dget_space(dataset);
  assert(dataspace >= 0);

  return dataspace;
}

void close_dataspace(hid_t dataspace) {
  if (dataspace >= 0) {
    assert(H5Sclose(dataspace) >= 0);
  }
}

/** other HDF5 **/

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

/** chromosome functions **/

void init_chromosome(chromosome_t *chromosome) {
  chromosome->chrom = xmalloc(sizeof(char));
  *(chromosome->chrom) = '\0';
  chromosome->h5file = -1;
}

void init_supercontig_array(size_t num_supercontigs, chromosome_t *chromosome)
{
  chromosome->num_supercontigs = num_supercontigs;
  chromosome->supercontigs = xmalloc(num_supercontigs * sizeof(supercontig_t));
  chromosome->supercontig_curr = chromosome->supercontigs;
}

herr_t supercontig_visitor(hid_t g_id, const char *name,
                           const H5L_info_t *info,
                           void *op_info) {
  hid_t subgroup;
  supercontig_t *supercontig;
  chromosome_t *chromosome;

  chromosome = (chromosome_t *) op_info;

  supercontig = chromosome->supercontig_curr++;

  /* leave open */
  subgroup = H5Gopen(g_id, name, H5P_DEFAULT);
  assert(subgroup >= 0);

  get_attr(subgroup, ATTR_START, H5T_STD_I32LE, &supercontig->start);
  get_attr(subgroup, ATTR_END, H5T_STD_I32LE, &supercontig->end);
  supercontig->group = subgroup;

  return 0;
}

supercontig_t *last_supercontig(chromosome_t *chromosome) {
  return chromosome->supercontigs + chromosome->num_supercontigs - 1;
}

void open_chromosome(chromosome_t *chromosome, const char *h5filename) {
  hid_t root = -1;
  H5G_info_t root_info;

  /* must be specified to H5Literate; allows interruption and
     resumption, but I don't use it */
  hsize_t idx = 0;

  /* open the chromosome file */
  chromosome->h5file = H5Fopen(h5filename, H5F_ACC_RDWR, H5P_DEFAULT);
  assert(chromosome->h5file >= 0);

  /* open the root group */
  root = H5Gopen(chromosome->h5file, "/", H5P_DEFAULT);
  assert(root >= 0);

  /* allocate supercontig metadata array */
  assert(H5Gget_info(root, &root_info) >= 0);
  init_supercontig_array(root_info.nlinks, chromosome);

  /* populate supercontig metadata array */
  assert(H5Literate(root, H5_INDEX_NAME, H5_ITER_INC, &idx,
                    supercontig_visitor, chromosome) == 0);

  assert(H5Gclose(root) >= 0);

}

void close_chromosome(chromosome_t *chromosome) {
  free(chromosome->chrom);

  if (chromosome->h5file < 0) {
    return;
  }

  for (supercontig_t *supercontig = chromosome->supercontigs;
       supercontig <= last_supercontig(chromosome); supercontig++) {
    assert(H5Gclose(supercontig->group) >= 0);
  }
  free(chromosome->supercontigs);

  close_file(chromosome->h5file);
}

/** specific auxiliary functions **/

/* fetch num_cols and the col for a particular trackname */
void get_cols(chromosome_t *chromosome, char *trackname, hsize_t *num_cols,
              hsize_t *col) {
  hid_t attr, root, dataspace, datatype;
  hsize_t data_size, cell_size, num_cells;
  char *attr_data;

  root = H5Gopen(chromosome->h5file, "/", H5P_DEFAULT);
  assert(root >= 0);

  attr = H5Aopen_name(root, "tracknames");
  assert(attr >= 0);

  dataspace = H5Aget_space(attr);
  assert(dataspace >= 0);

  assert(H5Sget_simple_extent_dims(dataspace, num_cols, NULL) == 1);
  assert(H5Sclose(dataspace) >= 0);

  if (trackname && col) {
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
  }

  assert(H5Aclose(attr) >= 0);
  assert(H5Gclose(root) >= 0);
}

/* make existing dataset into a PyTables CArray by setting appropriate
   attrs */
void make_pytables_carray(hid_t dataset) {
  set_attr_str(dataset, "CLASS", "CARRAY");
  set_attr_str(dataset, "TITLE", "");
  set_attr_str(dataset, "VERSION", "1.0");
}

hid_t open_supercontig_dataset(supercontig_t *supercontig, hsize_t num_cols) {
  /* creates it if it doesn't already exist;
     returns a handle for H5Dread or H5Dwrite */

  hid_t dataset = -1;
  hid_t dataset_creation_plist = -1;
  hid_t file_dataspace = -1;

  hsize_t file_dataspace_dims[CARDINALITY];
  hsize_t chunk_dims[CARDINALITY] = {CHUNK_NROWS, 1};

  /* set up creation options */
  dataset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);
  assert(dataset_creation_plist >= 0);

  assert(H5Pset_fill_value(dataset_creation_plist, DTYPE, &nan_float) >= 0);

  /* open dataset if it already exists */
  dataset = open_dataset(supercontig->group, DATASET_NAME, H5P_DEFAULT);

  if (dataset < 0) {
    /* create dataspace */
    file_dataspace_dims[0] = supercontig->end - supercontig->start;
    file_dataspace_dims[1] = num_cols;

    file_dataspace = H5Screate_simple(CARDINALITY, file_dataspace_dims, NULL);
    assert(file_dataspace >= 0);

    /* create chunkspace */
    assert(H5Pset_chunk(dataset_creation_plist, CARDINALITY, chunk_dims) >= 0);

    /* create dataset */
    fprintf(stderr, " creating %lld x %lld dataset...",
            file_dataspace_dims[0], file_dataspace_dims[1]);
    dataset = H5Dcreate(supercontig->group, DATASET_NAME, DTYPE,
                        file_dataspace, H5P_DEFAULT, dataset_creation_plist,
                        H5P_DEFAULT);
    assert(dataset >= 0);
    fprintf(stderr, " done\n");

    make_pytables_carray(dataset);
  }

  /* XXX: set dirty attribute */
  /* XXX: need to look up in documentation: boolean attributes, etc. */

  assert(H5Pclose(dataset_creation_plist) >= 0);

  return dataset;
}

hid_t get_col_dataspace(hsize_t *dims) {
  hid_t dataspace;

  dataspace = H5Screate_simple(1, dims, NULL);
  assert(dataspace >= 0);

  return dataspace;
}

/** general parsing **/

file_format sniff_header_line(const char *line) {
  if (!strncmp(ID_WIGFIX, line, strlen(ID_WIGFIX))) {
    return FMT_WIGFIX;
  } else if (!strncmp(ID_WIGVAR, line, strlen(ID_WIGVAR))) {
    return FMT_WIGVAR;
  }

  fatal("only fixedStep and variableStep formats supported");
  /* return FMT_BED; */

  return -1;
}

void parse_wiggle_header(char *line, file_format fmt, char **chrom,
                         long *start, long *step, long *span) {
  /* mallocs chrom; caller must free() it */
  /* start and step may be null pointers */

  char *save_ptr;
  char *token;
  char *tailptr;
  char *newstring;
  char *id_str;

  char *loc_eq;
  char *key;
  char *val;

  switch (fmt) {
  case FMT_WIGFIX:
    id_str = ID_WIGFIX;
    break;
  case FMT_WIGVAR:
    id_str = ID_WIGVAR;
    break;
  default:
    fprintf(stderr, "unsupported format: %d", fmt);
    exit(EXIT_FAILURE);
  }

  assert(!strncmp(id_str, line, strlen(id_str)));

  /* strip trailing newline */
  *strchr(line, '\n') = '\0';

  save_ptr = strdupa(line);
  assert(save_ptr);

  newstring = line + strlen(id_str);

  /* set to defaults */
  *span = 1;
  if (start) {
    *start = 1;
  }
  if (step) {
    *step = 1;
  }

  while ((token = strtok_r(newstring, DELIM_WIG, &save_ptr))) {
    loc_eq = strchr(token, '=');
    key = strndupa(token, loc_eq - token);
    assert(key);

    val = loc_eq + 1;

    errno = 0;

    if (!strcmp(key, KEY_CHROM)) {
      *chrom = strdup(val);
      assert(*chrom);

    } else if (!strcmp(key, KEY_START)) {
      assert(start); /* don't write a null pointer */

      /* correct 1-based coordinate */
      *start = strtol(val, &tailptr, BASE) - 1;
      assert(!errno && !*tailptr);

    } else if (!strcmp(key, KEY_STEP)) {
      assert(step); /* don't write a null pointer */
      *step = strtol(val, &tailptr, BASE);
      assert(!errno && !*tailptr);

    } else if (!strcmp(key, KEY_SPAN)) {
      *span = strtol(val, &tailptr, BASE);
      assert(!errno && !*tailptr);

    } else {
      fprintf(stderr, "can't understand key: %s\n", key);
      exit(EXIT_FAILURE);
    }

    newstring = NULL;
  }
}

/** writing **/

bool has_data(float *buf_start, float *buf_end) {
  /* check that there is even one data point in the supercontig
     offset region */
  for (float *buf_ptr = buf_start; buf_ptr < buf_end; buf_ptr++) {
    if (!isnan(*buf_ptr)) {
      return true;
    };
  }

  return false;
}

void write_buf(chromosome_t *chromosome, char *trackname,
               float *buf_start, float *buf_end,
               float *buf_filled_start, float *buf_filled_end) {
  float *buf_supercontig_start, *buf_supercontig_end;

  size_t start_offset, end_offset;
  hid_t dataset;

  hid_t mem_dataspace;
  hid_t file_dataspace = -1;

  hsize_t num_cols, col;

  hsize_t mem_dataspace_dims[CARDINALITY] = {-1, 1};
  hsize_t select_start[CARDINALITY];

  /* correct for overshoot */
  if (*buf_filled_end > *buf_end) {
    *buf_filled_end = *buf_end;
  }

  for (supercontig_t *supercontig = chromosome->supercontigs;
       supercontig <= last_supercontig(chromosome); supercontig++) {
    /* find the start that fits into this supercontig */
    start_offset = buf_filled_start - buf_start;
    if (start_offset < supercontig->start) {
      /* truncate */
      buf_supercontig_start = buf_start + supercontig->start;
      start_offset = supercontig->start;
    } else {
      buf_supercontig_start = buf_filled_start;
    }
    if (start_offset >= supercontig->end) {
      /* beyond the range of this supercontig */
      continue;
    }

    /* find the end that fits into this supercontig */
    end_offset = buf_filled_end - buf_start;
    if (end_offset > supercontig->end) {
      /* truncate */
      buf_supercontig_end = buf_start + supercontig->end;
      end_offset = supercontig->end;
    } else {
      buf_supercontig_end = buf_filled_end;
    }
    if (end_offset < supercontig->start) {
      continue;
    }

    assert(start_offset >= supercontig->start
           && end_offset <= supercontig->end
           && end_offset > start_offset);

    /* check for at least one data point */
    if (!has_data(buf_supercontig_start, buf_supercontig_end)) {
      continue;
    }

    /* set mem dataspace */
    mem_dataspace_dims[0] = end_offset - start_offset;
    mem_dataspace = get_col_dataspace(mem_dataspace_dims);

    /* calc dimensions */
    get_cols(chromosome, trackname, &num_cols, &col);

    /* open or create dataset */
    dataset = open_supercontig_dataset(supercontig, num_cols);

    /* get file dataspace */
    file_dataspace = get_file_dataspace(dataset);

    /* select file hyperslab */
    select_start[0] = start_offset - supercontig->start;
    select_start[1] = col;

    /* count has same dims as mem_dataspace */
    assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                               NULL, mem_dataspace_dims, NULL) >= 0);

    /* write */
    fprintf(stderr, " writing %lld floats...", mem_dataspace_dims[0]);
    assert(H5Dwrite(dataset, DTYPE, mem_dataspace, file_dataspace,
                    H5P_DEFAULT, buf_supercontig_start) >= 0);
    fprintf(stderr, " done\n");

    /* close all */
    close_dataset(dataset);
    close_dataspace(file_dataspace);
    close_dataspace(mem_dataspace);
  }
}

void seek_chromosome(char *chrom, char *h5dirname, chromosome_t *chromosome) {
  char *h5filename = NULL;
  char *h5filename_suffix;

  fprintf(stderr, "%s\n", chrom);

  /* allocate space for h5filename, including 2 extra bytes for '/' and '\0' */
  h5filename = alloca(strlen(h5dirname) + strlen(chrom) + strlen(SUFFIX_H5)
                      + 2);
  assert(h5filename);

  /* set h5filename */
  h5filename_suffix = stpcpy(h5filename, h5dirname);
  h5filename_suffix = stpcpy(h5filename_suffix, "/");
  h5filename_suffix = stpcpy(h5filename_suffix, chrom);
  strcpy(h5filename_suffix, SUFFIX_H5);

  close_chromosome(chromosome);
  open_chromosome(chromosome, h5filename);

  chromosome->chrom = chrom;
}

void malloc_chromosome_buf(chromosome_t *chromosome,
                           float **buf_start, float **buf_end) {
  /* allocate enough space to assign values from 0 to the end of the
     last supercontig */

  size_t buf_len;

  /* XXX: last_supercontig(chromosome) might not return the maximum
     value; you really need to iterate through all of them */
  buf_len = last_supercontig(chromosome)->end;

  if (*buf_start) {
    free(*buf_start);
  }
  *buf_start = xmalloc(buf_len * sizeof(float));
  *buf_end = *buf_start + buf_len;
}

/** wigFix **/

void proc_wigfix_header(char *line, char *h5dirname, chromosome_t *chromosome,
                        float **buf_start, float **buf_end, float **fill_start,
                        long *step, long *span) {
  long start = -1;

  char *chrom = NULL;

  /* do writing if buf_len > 0 */
  parse_wiggle_header(line, FMT_WIGFIX, &chrom, &start, step, span);
  assert(chrom && start >= 0 && *step >= 1 && *span >= 1);

  /* chromosome->chrom is always initialized, at least to NULL, and
     chrom is never NULL */
  if (strcmp(chrom, chromosome->chrom)) {
    /* only reseek and malloc if it is different */
    seek_chromosome(chrom, h5dirname, chromosome);
    malloc_chromosome_buf(chromosome, buf_start, buf_end);
  }

  *fill_start = *buf_start + start;
}

void proc_wigfix(char *h5dirname, char *trackname, char *line,
                 size_t *size_line) {
  char *tailptr;

  float *buf_start = NULL;
  /* buf_filled_start is overall fill start; fill_start is current fill start*/
  float *buf_filled_start, *fill_start, *fill_end, *buf_end;

  chromosome_t chromosome;

  long step = 1;
  long span = 1;
  float datum;

  init_chromosome(&chromosome);

  proc_wigfix_header(line, h5dirname, &chromosome,
                     &buf_start, &buf_end, &fill_start, &step, &span);

  buf_filled_start = fill_start;

  while (getline(&line, size_line, stdin) >= 0) {
    errno = 0;
    datum = strtof(line, &tailptr);
    assert(!errno);

    if (*tailptr == '\n') {
      if (fill_start < buf_end) {
        fill_end = fill_start + span;
        if (fill_end > buf_end) {
          fprintf(stderr, " ignoring data at %s:%ld+%ld\n",
                  chromosome.chrom, fill_start - buf_start, span);
          fill_end = buf_end;
        }

        /* write into buffer */
        for (float *buf_ptr = fill_start; buf_ptr < fill_end; buf_ptr++) {
          *buf_ptr = datum;
        }

        fill_start += step;
      } else {
        /* else: ignore data until we get to another header line */
        fprintf(stderr, " ignoring data at %s:%ld\n",
                chromosome.chrom, fill_start - buf_start);
      }
    } else {
      write_buf(&chromosome, trackname, buf_start, buf_end,
                buf_filled_start, fill_start);
      proc_wigfix_header(line, h5dirname, &chromosome,
                         &buf_start, &buf_end, &fill_start, &step, &span);
      buf_filled_start = fill_start;
    }
  }

  write_buf(&chromosome, trackname, buf_start, buf_end,
            buf_filled_start, fill_start);

  close_chromosome(&chromosome);
  free(buf_start);
}

/** wigVar **/

void proc_wigvar_header(char *line, char *h5dirname, chromosome_t *chromosome,
                        char *trackname, float **buf_start, float **buf_end,
                        long *span) {
  char *chrom = NULL;

  hid_t mem_dataspace, file_dataspace;
  hid_t dataset;

  hsize_t num_cols, col;

  hsize_t mem_dataspace_dims[CARDINALITY] = {-1, 1};
  hsize_t select_start[CARDINALITY];

  /* do writing if buf_len > 0 */
  parse_wiggle_header(line, FMT_WIGVAR, &chrom, NULL, NULL, span);
  assert(chrom && *span >= 1);

  /* chromosome->chrom is always initialized, at least to NULL, and
     chrom is never NULL */
  if (strcmp(chrom, chromosome->chrom)) {
    /* only reseek and malloc if it is different */
    /* XXX: should probably be an assertion rather than an if */
    seek_chromosome(chrom, h5dirname, chromosome);
    malloc_chromosome_buf(chromosome, buf_start, buf_end);

    /* clear memory */
    for (float *buf_ptr = *buf_start; buf_ptr < *buf_end; buf_ptr++) {
      *buf_ptr = NAN;
    }

    /* calc dimensions */
    get_cols(chromosome, trackname, &num_cols, &col);

    for (supercontig_t *supercontig = chromosome->supercontigs;
         supercontig <= last_supercontig(chromosome); supercontig++) {
      /* set mem dataspace */
      mem_dataspace_dims[0] = supercontig->end - supercontig->start;
      mem_dataspace = get_col_dataspace(mem_dataspace_dims);

      /* open or create dataset */
      dataset = open_supercontig_dataset(supercontig, num_cols);

      /* get file dataspace */
      file_dataspace = get_file_dataspace(dataset);

      /* select file hyperslab */
      select_start[0] = 0;
      select_start[1] = col;

      /* count has same dims as mem_dataspace */
      assert(H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, select_start,
                                 NULL, mem_dataspace_dims, NULL) >= 0);

      /* read */
      fprintf(stderr, " reading %lld floats...", mem_dataspace_dims[0]);
      assert(H5Dread(dataset, DTYPE, mem_dataspace, file_dataspace,
                     H5P_DEFAULT, (*buf_start) + supercontig->start) >= 0);
      fprintf(stderr, " done\n");

      /* close all */
      close_dataset(dataset);
      close_dataspace(file_dataspace);
      close_dataspace(mem_dataspace);
    }
  }
}

void proc_wigvar(char *h5dirname, char *trackname, char *line,
                 size_t *size_line) {
  char *tailptr;

  float *buf_start = NULL;
  float *buf_end;
  float *fill_start, *fill_end;
  float datum;

  chromosome_t chromosome;

  long start;
  long span = 1;

  init_chromosome(&chromosome);

  proc_wigvar_header(line, h5dirname, &chromosome, trackname,
                     &buf_start, &buf_end, &span);

  while (getline(&line, size_line, stdin) >= 0) {
    /* correcting 1-based coordinate */
    errno = 0;
    start = strtol(line, &tailptr, BASE) - 1;
    assert(!errno);

    /* next char must be space */
    if (tailptr != line && isblank(*tailptr)) {
      assert(start >= 0);

      errno = 0;
      datum = strtof(tailptr, &tailptr);
      assert(!errno);

      /* must be EOL */
      assert(*tailptr == '\n');

      fill_start = buf_start + start;
      if (fill_start > buf_end) {
        fprintf(stderr, " ignoring data at %s:%ld\n", chromosome.chrom, start);
        continue;
      }

      fill_end = fill_start + span;
      if (fill_end > buf_end) {
        fprintf(stderr, " ignoring data at %s:%ld+%ld\n",
                chromosome.chrom, start, span);
        fill_end = buf_end;
      }

      /* write into buffer */
      for (float *buf_ptr = fill_start; buf_ptr < fill_end; buf_ptr++) {
        *buf_ptr = datum;
      }

    } else {
      write_buf(&chromosome, trackname, buf_start, buf_end,
                buf_start, buf_end);
      proc_wigvar_header(line, h5dirname, &chromosome, trackname,
                         &buf_start, &buf_end, &span);
    }
  }

  write_buf(&chromosome, trackname, buf_start, buf_end, buf_start, buf_end);

  close_chromosome(&chromosome);
  free(buf_start);
}

/** programmatic interface **/

void load_data(char *h5dirname, char *trackname) {
  char *line = NULL;
  size_t size_line = 0;

  file_format fmt;

  /* XXXopt: would be faster to just read a big block and do repeated
     strtof rather than using getline */

  if (getline(&line, &size_line, stdin) < 0) {
    error_at_line(EXIT_FAILURE, errno, __FILE__, __LINE__,
                  "failed to read first line");
  }

  fmt = sniff_header_line(line);

  /* XXX: allow mixing and matching later on. for now, once you pick a
     format, you are stuck */
  switch (fmt) {
  case FMT_WIGFIX:
    proc_wigfix(h5dirname, trackname, line, &size_line);
    break;
  case FMT_WIGVAR:
    proc_wigvar(h5dirname, trackname, line, &size_line);
    break;
  case FMT_BED:
  default:
    fatal("only fixedStep and variableStep formats supported");
    break;
  }

  /* free heap variables */
  free(line);
}

/** command-line interface **/

const char *argp_program_version = "$Revision$";
const char *argp_program_bug_address = "Michael Hoffman <mmh1@washington.edu>";

static char doc[] = "Loads data into genomedata format";
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

