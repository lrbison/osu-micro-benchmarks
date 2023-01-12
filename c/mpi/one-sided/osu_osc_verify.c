#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <mpi.h>
#include <string.h>

#include "osu_util.h"

// not consistent with errno.h, but errno.h conflicts with CUDA_CHECK macro.
#define ENODATA 2

static int set_hmem_buffer (void *dst, void *src, size_t size)
{
	// return;
    switch (options.accel) {
	case NONE:
		memcpy(dst, src, size);
		break;
#ifdef _ENABLE_CUDA_
        case CUDA:
		CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaDeviceSynchronize());
		break;
#endif
	default:
		printf("Memory copy not implemented for the selected acceleration platform\n");
		return -1;
    }
    return 0;
}

static int get_hmem_buffer (void *dst, void *src, size_t size)
{
	// return;
    switch (options.accel) {
	case NONE:
		memcpy(dst, src, size);
		break;
#ifdef _ENABLE_CUDA_
        case CUDA:
		CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaDeviceSynchronize());
		break;
#endif
	default:
		printf("Memory copy not implemented for the selected acceleration platform\n");
		return -1;
    }
    return 0;
}
struct atomic_dv_summary {
	MPI_Datatype datatype;
	MPI_Op op;
	size_t trials;
	size_t validation_failures;
	size_t validations_performed;
	size_t first_failure;
	size_t last_failure;
	struct atomic_dv_summary *next;
};

#define bool _Bool
#define MPI_OP_COUNT 16

struct atomic_dv_summary* dv_summary_root = NULL;

typedef enum type_of_enum {
    FI_TYPE_ATOMIC_TYPE,
    FI_TYPE_ATOMIC_OP
} type_of_enum;

char fi_str_output[256];
char* fi_tostr(void *val, type_of_enum type_type) {
	memset(fi_str_output, 0, sizeof(fi_str_output));
	if (type_type == FI_TYPE_ATOMIC_OP) {
		MPI_Op op = *(MPI_Op*)val;
		if (-1 == *(int*)val)	sprintf(fi_str_output,"%s", "Compare-And-Swap");
		if (op == MPI_OP_NULL)  sprintf(fi_str_output,"%s", "MPI_OP_NULL");
		if (op == MPI_MAX)      sprintf(fi_str_output,"%s", "MPI_MAX");
		if (op == MPI_MIN)      sprintf(fi_str_output,"%s", "MPI_MIN");
		if (op == MPI_SUM)      sprintf(fi_str_output,"%s", "MPI_SUM");
		if (op == MPI_PROD)     sprintf(fi_str_output,"%s", "MPI_PROD");
		if (op == MPI_LAND)	sprintf(fi_str_output,"%s", "MPI_LAND");
		if (op == MPI_BAND)     sprintf(fi_str_output,"%s", "MPI_BAND");
		if (op == MPI_LOR)      sprintf(fi_str_output,"%s", "MPI_LOR");
		if (op == MPI_BOR)      sprintf(fi_str_output,"%s", "MPI_BOR");
		if (op == MPI_LXOR)     sprintf(fi_str_output,"%s", "MPI_LXOR");
		if (op == MPI_BXOR)     sprintf(fi_str_output,"%s", "MPI_BXOR");
		if (op == MPI_MINLOC)	sprintf(fi_str_output,"%s", "MPI_MINLOC");
		if (op == MPI_MAXLOC)   sprintf(fi_str_output,"%s", "MPI_MAXLOC");
		if (op == MPI_REPLACE)  sprintf(fi_str_output,"%s", "MPI_REPLACE");
	}
	if (type_type == FI_TYPE_ATOMIC_TYPE) {
		MPI_Datatype type = *(MPI_Datatype*)val;
		if (type == MPI_DATATYPE_NULL)		sprintf(fi_str_output,"%s", "MPI_DATATYPE_NULL");
		if (type == MPI_SIGNED_CHAR)		sprintf(fi_str_output,"%s", "MPI_SIGNED_CHAR");
		if (type == MPI_UNSIGNED_CHAR)		sprintf(fi_str_output,"%s", "MPI_UNSIGNED_CHAR");
		if (type == MPI_SHORT)			sprintf(fi_str_output,"%s", "MPI_SHORT");
		if (type == MPI_UNSIGNED_SHORT)		sprintf(fi_str_output,"%s", "MPI_UNSIGNED_SHORT");
		if (type == MPI_INT)			sprintf(fi_str_output,"%s", "MPI_INT");
		if (type == MPI_UNSIGNED)		sprintf(fi_str_output,"%s", "MPI_UNSIGNED");
		if (type == MPI_LONG)			sprintf(fi_str_output,"%s", "MPI_LONG");
		if (type == MPI_UNSIGNED_LONG)  	sprintf(fi_str_output,"%s", "MPI_UNSIGNED_LONG");
		if (type == MPI_LONG_LONG)      	sprintf(fi_str_output,"%s", "MPI_LONG_LONG");
		if (type == MPI_UNSIGNED_LONG_LONG)     sprintf(fi_str_output,"%s", "MPI_UNSIGNED_LONG_LONG");
		if (type == MPI_FLOAT)     		sprintf(fi_str_output,"%s", "MPI_FLOAT");
		if (type == MPI_DOUBLE)			sprintf(fi_str_output,"%s", "MPI_DOUBLE");
		if (type == MPI_LONG_DOUBLE)   		sprintf(fi_str_output,"%s", "MPI_LONG_DOUBLE");
		if (type == MPI_C_FLOAT_COMPLEX)	sprintf(fi_str_output,"%s", "MPI_C_FLOAT_COMPLEX");
		if (type == MPI_C_DOUBLE_COMPLEX) 	sprintf(fi_str_output,"%s", "MPI_C_DOUBLE_COMPLEX");
		if (type == MPI_C_LONG_DOUBLE_COMPLEX)	sprintf(fi_str_output,"%s", "MPI_C_LONG_DOUBLE_COMPLEX");
	}
	return fi_str_output;
}

int mpi_op_enumerate(MPI_Op op) {
    if (op == MPI_OP_NULL)  return 0;
    if (op == MPI_MAX)      return 1;
    if (op == MPI_MIN)      return 2;
    if (op == MPI_SUM)      return 3;
    if (op == MPI_PROD)     return 4;
    if (op == MPI_LAND)     return 5;
    if (op == MPI_BAND)     return 6;
    if (op == MPI_LOR)      return 7;
    if (op == MPI_BOR)      return 8;
    if (op == MPI_LXOR)     return 9;
    if (op == MPI_BXOR)     return 10;
    if (op == MPI_MINLOC)   return 11;
    if (op == MPI_MAXLOC)   return 12;
    if (op == MPI_REPLACE)  return 13;
    return -1;
}
#define ENUM_OF_DMPI_OP_NULL	0
#define ENUM_OF_DMPI_MAX	1
#define ENUM_OF_DMPI_MIN	2
#define ENUM_OF_DMPI_SUM	3
#define ENUM_OF_DMPI_PROD	4
#define ENUM_OF_DMPI_LAND	5
#define ENUM_OF_DMPI_BAND	6
#define ENUM_OF_DMPI_LOR	7
#define ENUM_OF_DMPI_BOR	8
#define ENUM_OF_DMPI_LXOR	9
#define ENUM_OF_DMPI_BXOR	10
#define ENUM_OF_DMPI_MINLOC	11
#define ENUM_OF_DMPI_MAXLOC	12
#define ENUM_OF_DMPI_REPLACE	13

int mpi_dtype_enumerate(MPI_Datatype dtype) {
  if (dtype == MPI_DATATYPE_NULL)       return 0;
  if (dtype == MPI_SIGNED_CHAR)         return 1;
  if (dtype == MPI_UNSIGNED_CHAR)       return 2;
  if (dtype == MPI_SHORT)               return 3;
  if (dtype == MPI_UNSIGNED_SHORT)      return 4;
  if (dtype == MPI_INT)                 return 5;
  if (dtype == MPI_UNSIGNED)            return 6;
  if (dtype == MPI_LONG)                return 7;
  if (dtype == MPI_UNSIGNED_LONG)       return 8;
  if (dtype == MPI_LONG_LONG)           return 9;
  if (dtype == MPI_UNSIGNED_LONG_LONG)  return 10;
  if (dtype == MPI_FLOAT)               return 11;
  if (dtype == MPI_DOUBLE)              return 12;
  if (dtype == MPI_LONG_DOUBLE)         return 13;
  if (dtype == MPI_C_FLOAT_COMPLEX)     return 14;
  if (dtype == MPI_C_DOUBLE_COMPLEX)    return 15;
  if (dtype == MPI_C_LONG_DOUBLE_COMPLEX) return 16;
  return -1;
}
// these must match the above routine!
#define ENUM_OF_DMPI_DATATYPE_NULL 	0
#define ENUM_OF_DMPI_SIGNED_CHAR	1
#define ENUM_OF_DMPI_UNSIGNED_CHAR	2
#define ENUM_OF_DMPI_SHORT		3
#define ENUM_OF_DMPI_UNSIGNED_SHORT	4
#define ENUM_OF_DMPI_INT		5
#define ENUM_OF_DMPI_UNSIGNED		6
#define ENUM_OF_DMPI_LONG		7
#define ENUM_OF_DMPI_UNSIGNED_LONG	8
#define ENUM_OF_DMPI_LONG_LONG		9
#define ENUM_OF_DMPI_UNSIGNED_LONG_LONG	10
#define ENUM_OF_DMPI_FLOAT		11
#define ENUM_OF_DMPI_DOUBLE		12
#define ENUM_OF_DMPI_LONG_DOUBLE	13
#define ENUM_OF_DMPI_C_FLOAT_COMPLEX	14
#define ENUM_OF_DMPI_C_DOUBLE_COMPLEX	15
#define ENUM_OF_DMPI_C_LONG_DOUBLE_COMPLEX 16

/*
 @brief Prints a summary of test failures.
 @return 0 if all validations passed, <0 if any failures recorded.
*/
int atomic_data_validation_print_summary() {

	int retval = 0;
	char type_str[32] = {0};
	char op_str[32] = {0};
	char test_name[64] = {};
	int validation_combos = 0;
	int failure_count = 0;

	struct atomic_dv_summary *node = dv_summary_root;
	struct atomic_dv_summary *next = NULL;

	if (!node) {
		printf("SKIPPED: No validations were performed!\n");
		return 0;
	}

	while(node) {
		snprintf(type_str, sizeof(type_str)-1, "%s", fi_tostr(&node->datatype, FI_TYPE_ATOMIC_TYPE));
		snprintf(op_str, sizeof(op_str)-1, "%s", fi_tostr(&node->op, FI_TYPE_ATOMIC_OP));
		snprintf(test_name, sizeof(test_name), "%s on %s", op_str, type_str);
		validation_combos += 1;

		if (node->validation_failures==0 && node->validations_performed==node->trials) {
			// all these tests passed
			//printf("PASSED: %s passed %zu trials.\n",test_name, node->trials);
		}
		else if (node->validation_failures) {
			printf("FAILED: %s had %zu of %zu tests fail data validation.\n",
				test_name, node->validation_failures, node->trials);
			printf("\t\tFirst failure at trial %zu, last failure at trial %zu.\n",
				node->first_failure, node->last_failure);
			retval = -1;
			failure_count++;
		}
		else if (node->validations_performed < node->trials) {
			printf("SKIPPED: Data validation not available for %s\n", test_name);
			retval = -1;
			failure_count++;
		}

		// clean up as we go
		next = node->next;
		free(node);
		node = next;
	}

	if (retval == 0) {
		printf("PASSED: All %d combinations of ops and datatypes tested passed.\n",validation_combos);
	} else {
		printf("FAILED: %d of the %d combinations of ops and datatypes tested failed.\n",failure_count,validation_combos);
	}
	dv_summary_root = NULL;
	return retval;
}

static void atomic_dv_record(MPI_Datatype dtype, MPI_Op op, bool failed, bool checked) {

	struct atomic_dv_summary *node = dv_summary_root;

	if (!node || node->op != op || node->datatype != dtype) {
		// allocate and add a new node
		node = calloc(1, sizeof(struct atomic_dv_summary));
		node->next = dv_summary_root;
		dv_summary_root = node;
		node->op = op;
		node->datatype = dtype;
	}

	// record trial.
	node->trials++;
	if (failed) {
		if (node->validation_failures==0) node->first_failure = node->trials;
		node->last_failure = node->trials;
		node->validation_failures++;
	}
	if (checked) node->validations_performed++;
}


// debugging macro help: gcc -Iinclude -I/fsx/lrbison/libfabric/install/include -E functional/atomic_verify.c | sed 's/case/\ncase/g' | less

#define ATOM_FOR_DMPI_MIN(a,ao,b)  (ao) = (((a) < (b)) ? a : b)
#define ATOM_FOR_DMPI_MAX(a,ao,b)  (ao) = (((a) > (b)) ? a : b)
#define ATOM_FOR_DMPI_SUM(a,ao,b)  (ao) = ((a) + (b))
#define ATOM_FOR_DMPI_PROD(a,ao,b) (ao) = ((a) * (b))
#define ATOM_FOR_DMPI_LOR(a,ao,b)  (ao) = ((a) || (b))
#define ATOM_FOR_DMPI_LAND(a,ao,b) (ao) = ((a) && (b))
#define ATOM_FOR_DMPI_BOR(a,ao,b)  (ao) = ((a) | (b))
#define ATOM_FOR_DMPI_BAND(a,ao,b) (ao) = ((a) & (b))
#define ATOM_FOR_DMPI_LXOR(a,ao,b) (ao) = (((a) && !(b)) || (!(a) && (b)))
#define ATOM_FOR_DMPI_BXOR(a,ao,b) (ao) = ((a) ^ (b))
#define ATOM_FOR_DMPI_ATOMIC_READ(a,ao,b)  (ao) = (a)
#define ATOM_FOR_DMPI_ATOMIC_WRITE(a,ao,b) (ao) = (b)

#define ATOM_FOR_DMPI_CSWAP(a,ao,b,c)    if ((c) == (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_CSWAP_NE(a,ao,b,c) if ((c) != (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_CSWAP_LE(a,ao,b,c) if ((c) <= (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_CSWAP_LT(a,ao,b,c) if ((c) <  (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_CSWAP_GE(a,ao,b,c) if ((c) >= (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_CSWAP_GT(a,ao,b,c) if ((c) >  (a)) {(ao) = (b);}
#define ATOM_FOR_DMPI_MSWAP(a,ao,b,c)    (ao) = ((b) & (c)) | ((a) & ~(c));

#define ATOM_FOR_CPLX_DMPI_MIN(a,ao,b,absfun)  (ao) = ((absfun(a) < absfun(b)) ? (a) : (b))
#define ATOM_FOR_CPLX_DMPI_MAX(a,ao,b,absfun)  (ao) = (absfun(a) > absfun(b) ? (a) : (b))
#define ATOM_FOR_CPLX_DMPI_CSWAP_LE(a,ao,b,c,absfun) if (absfun(c) <= absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_DMPI_CSWAP_LT(a,ao,b,c,absfun) if (absfun(c) <  absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_DMPI_CSWAP_GE(a,ao,b,c,absfun) if (absfun(c) >= absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_DMPI_CSWAP_GT(a,ao,b,c,absfun) if (absfun(c) >  absfun(a)) {(ao) = (b);}

#define ATOM_CTYPE_FOR_DMPI_SIGNED_CHAR char
#define ATOM_CTYPE_FOR_DMPI_UNSIGNED_CHAR unsigned char
#define ATOM_CTYPE_FOR_DMPI_SHORT short
#define ATOM_CTYPE_FOR_DMPI_UNSIGNED_SHORT unsigned short
#define ATOM_CTYPE_FOR_DMPI_INT int
#define ATOM_CTYPE_FOR_DMPI_UNSIGNED unsigned
#define ATOM_CTYPE_FOR_DMPI_LONG long
#define ATOM_CTYPE_FOR_DMPI_UNSIGNED_LONG unsigned long
#define ATOM_CTYPE_FOR_DMPI_LONG_LONG long long
#define ATOM_CTYPE_FOR_DMPI_UNSIGNED_LONG_LONG unsigned long long

#define ATOM_CTYPE_FOR_DMPI_FLOAT float
#define ATOM_CTYPE_FOR_DMPI_DOUBLE double
#define ATOM_CTYPE_FOR_DMPI_LONG_DOUBLE long double

#define ATOM_CTYPE_FOR_DMPI_C_FLOAT_COMPLEX float complex
#define ATOM_CTYPE_FOR_DMPI_C_DOUBLE_COMPLEX double complex
#define ATOM_CTYPE_FOR_DMPI_C_LONG_DOUBLE_COMPLEX long double complex

// this macro is for expansion inside the perform_atomic_op function
// and uses variables local to that function.
#define atomic_case_cplx(ftype, fop, absfun)					\
case ftype*MPI_OP_COUNT + fop:				                	\
	{   if(result) *(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in;	\
		ATOM_FOR_CPLX_##fop( 	*(ATOM_CTYPE_FOR_##ftype*)addr_in,	\
					*(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
					*(ATOM_CTYPE_FOR_##ftype*)buf,		\
					absfun );				\
		break;								\
	}

#define atomic_case(ftype, fop)							\
case ENUM_OF_##ftype*MPI_OP_COUNT + ENUM_OF_##fop:				\
	{   if(result) *(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in;	\
		ATOM_FOR_##fop(	*(ATOM_CTYPE_FOR_##ftype*)addr_in,		\
				*(ATOM_CTYPE_FOR_##ftype*)addr_out,		\
				*(ATOM_CTYPE_FOR_##ftype*)buf );		\
		break;								\
	}


// this macro is for expansion inside the perform_atomic_cas function
// and uses variables local to that function.
#define atomic_case_cas(ftype)							\
case ENUM_OF_##ftype:									\
	{   if(result) {*(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in; }	\
		ATOM_FOR_DMPI_CSWAP(	*(ATOM_CTYPE_FOR_##ftype*)addr_in,	\
					*(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
					*(ATOM_CTYPE_FOR_##ftype*)buf,		\
					*(ATOM_CTYPE_FOR_##ftype*)compare );	\
		break;								\
	}

#define atomic_int_ops(dtype)				\
	atomic_case(dtype, DMPI_MIN)			\
	atomic_case(dtype, DMPI_MAX)			\
	atomic_case(dtype, DMPI_SUM)			\
	atomic_case(dtype, DMPI_PROD)			\
	atomic_case(dtype, DMPI_LOR)			\
	atomic_case(dtype, DMPI_LAND)			\
	atomic_case(dtype, DMPI_BOR)			\
	atomic_case(dtype, DMPI_BAND)			\
	atomic_case(dtype, DMPI_LXOR)			\
	atomic_case(dtype, DMPI_BXOR)
	// atomic_case_compare(dtype, FI_CSWAP)		\
	// atomic_case_compare(dtype, FI_CSWAP_NE)	\
	// atomic_case_compare(dtype, FI_CSWAP_LE)	\
	// atomic_case_compare(dtype, FI_CSWAP_LT)	\
	// atomic_case_compare(dtype, FI_CSWAP_GE)	\
	// atomic_case_compare(dtype, FI_CSWAP_GT)	\
	// atomic_case_compare(dtype, FI_MSWAP)


#define atomic_real_float_ops(dtype)		\
	atomic_case(dtype, DMPI_MIN)			\
	atomic_case(dtype, DMPI_MAX)			\
	atomic_case(dtype, DMPI_SUM)			\
	atomic_case(dtype, DMPI_PROD)			\
	atomic_case(dtype, DMPI_LOR)			\
	atomic_case(dtype, DMPI_LAND)			\
	atomic_case(dtype, DMPI_LXOR)			
	// atomic_case(dtype, FI_ATOMIC_READ)		\
	// atomic_case(dtype, FI_ATOMIC_WRITE)		\
	// atomic_case_compare(dtype, FI_CSWAP)	\
	// atomic_case_compare(dtype, FI_CSWAP_NE)	\
	// atomic_case_compare(dtype, FI_CSWAP_LE)	\
	// atomic_case_compare(dtype, FI_CSWAP_LT)	\
	// atomic_case_compare(dtype, FI_CSWAP_GE)	\
	// atomic_case_compare(dtype, FI_CSWAP_GT)

#define atomic_complex_float_ops(dtype, absfun)			\
	atomic_case(dtype, DMPI_SUM)				\
	atomic_case(dtype, DMPI_PROD)				
	// atomic_case_cplx(dtype, MPI_MIN, absfun)			
	// atomic_case_cplx(dtype, MPI_MAX, absfun)			
	
	// atomic_case(dtype, MPI_LOR)				\
	// atomic_case(dtype, MPI_LAND)				\
	// atomic_case(dtype, MPI_LXOR)				\
	// atomic_case(dtype, FI_ATOMIC_READ)			\
	// atomic_case(dtype, FI_ATOMIC_WRITE)			\
	// atomic_case_compare(dtype, FI_CSWAP)			\
	// atomic_case_compare(dtype, FI_CSWAP_NE)			\


int perform_atomic_op(	MPI_Datatype dtype,
			MPI_Op op,
			void *addr_in,
			void *buf,
			void *addr_out,
			void *compare,
			void *result)
{
    int op_enumeration = mpi_op_enumerate(op);
    int dtype_enumeration = mpi_dtype_enumerate(dtype);
	switch(dtype_enumeration*MPI_OP_COUNT + op_enumeration) {
		atomic_int_ops(DMPI_SIGNED_CHAR)
		atomic_int_ops(DMPI_UNSIGNED_CHAR)
		atomic_int_ops(DMPI_SHORT)
		atomic_int_ops(DMPI_UNSIGNED_SHORT)
		atomic_int_ops(DMPI_INT)
		atomic_int_ops(DMPI_UNSIGNED)
		atomic_int_ops(DMPI_LONG_LONG)
		atomic_int_ops(DMPI_UNSIGNED_LONG)
		atomic_int_ops(DMPI_UNSIGNED_LONG_LONG)

		atomic_real_float_ops(DMPI_FLOAT)
		atomic_real_float_ops(DMPI_DOUBLE)
		atomic_real_float_ops(DMPI_LONG_DOUBLE)

		atomic_complex_float_ops(DMPI_C_FLOAT_COMPLEX, cabsf)
		atomic_complex_float_ops(DMPI_C_DOUBLE_COMPLEX, cabs)
		atomic_complex_float_ops(DMPI_C_LONG_DOUBLE_COMPLEX, cabsl)

		default:
			return -1;

	}
	return 0;
}

int perform_atomic_cas(	MPI_Datatype dtype,
			void *addr_in,
			void *buf,
			void *addr_out,
			void *compare,
			void *result)
{
    	int dtype_enumeration = mpi_dtype_enumerate(dtype);
	switch(dtype_enumeration) {
		atomic_case_cas(DMPI_SIGNED_CHAR)
		atomic_case_cas(DMPI_UNSIGNED_CHAR)
		atomic_case_cas(DMPI_SHORT)
		atomic_case_cas(DMPI_UNSIGNED_SHORT)
		atomic_case_cas(DMPI_INT)
		atomic_case_cas(DMPI_UNSIGNED)
		atomic_case_cas(DMPI_LONG_LONG)
		atomic_case_cas(DMPI_UNSIGNED_LONG)
		atomic_case_cas(DMPI_UNSIGNED_LONG_LONG)
		atomic_case_cas(DMPI_FLOAT)
		atomic_case_cas(DMPI_DOUBLE)
		atomic_case_cas(DMPI_LONG_DOUBLE)

		default:
			return -1;

	}
	return 0;
}

static int validation_input_value(MPI_Datatype dtype, int jrank, void *val) {

	if (dtype == MPI_DATATYPE_NULL) {}
	else if (dtype == MPI_SIGNED_CHAR)
		*(char*)val = (1+jrank)*10;
	else if (dtype == MPI_UNSIGNED_CHAR)
		*(unsigned char*)val = (1+jrank)*10;
	else if (dtype == MPI_SHORT)
		*(short*)val = (1+jrank)*10;
	else if (dtype == MPI_UNSIGNED_SHORT)
		*(unsigned short*)val = (1+jrank)*10;
	else if (dtype == MPI_INT)
		*(int*)val = (1+jrank)*10;
	else if (dtype == MPI_UNSIGNED)
		*(unsigned*)val = (1+jrank)*10;
	else if (dtype == MPI_LONG)
		*(long*)val = (1+jrank)*10;
	else if (dtype == MPI_UNSIGNED_LONG)
		*(unsigned long*)val = (1+jrank)*10;
	else if (dtype == MPI_LONG_LONG)
		*(long long*)val = (1+jrank)*10;
	else if (dtype == MPI_UNSIGNED_LONG_LONG)
		*(unsigned long long*)val = (1+jrank)*10;
	else if (dtype == MPI_FLOAT)
		*(float*)val = (1+jrank)*1.11f;
	else if (dtype == MPI_DOUBLE)
		*(double*)val = (1+jrank)*1.11;
	else if (dtype == MPI_LONG_DOUBLE)
		*(long double*)val = (1+jrank)*1.11L;
	else if (dtype == MPI_C_FLOAT_COMPLEX)
		*(float complex*)val = CMPLXF( (1+jrank)*1.11f, (1+jrank*-0.5f) );
	else if (dtype == MPI_C_DOUBLE_COMPLEX) {
		*(double complex*)val = CMPLX( (1+jrank)*1.11, (1+jrank*-0.5) );
	}
	else if (dtype == MPI_C_LONG_DOUBLE_COMPLEX)
		*(long double complex*)val = CMPLXL( (1+jrank)*1.11L, (1+jrank*-0.5L) );
	else {
		fprintf(stderr, "No initial value defined, cannot perform data validation "
				"on atomic operations using %s\n",
			fi_tostr(&dtype, FI_TYPE_ATOMIC_TYPE) );
		return -1;
	}
	return 0;
}

#define COMPARE_AS_TYPE(c_type, a, b) *(c_type*)(a) == *(c_type*)(b)
static int atom_binary_compare(MPI_Datatype dtype, void *a, void *b)
{
	int dtype_size = 0;
	char *achar, *bchar;
	int err;

	// treat floating point types specially.  Avoid NaNs, since NaN != NaN.
	if (dtype == MPI_C_LONG_DOUBLE_COMPLEX) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_C_LONG_DOUBLE_COMPLEX, a, b);
	}
	if (dtype == MPI_C_DOUBLE_COMPLEX) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_C_DOUBLE_COMPLEX, a, b);
	}
	if (dtype == MPI_C_FLOAT_COMPLEX) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_C_FLOAT_COMPLEX, a, b);
	}
	if (dtype == MPI_LONG_DOUBLE) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_LONG_DOUBLE, a, b);
	}
	if (dtype == MPI_DOUBLE) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_DOUBLE, a, b);
	}
	if (dtype == MPI_FLOAT) {
		return COMPARE_AS_TYPE(ATOM_CTYPE_FOR_DMPI_FLOAT, a, b);
	}

	// treat remaining integers based soley on their size	
	err = MPI_Type_size(dtype, &dtype_size);
	if (err) return 0;

	switch (dtype_size)
	{
		case 1: return COMPARE_AS_TYPE(__int8_t, a, b);
		case 2: return COMPARE_AS_TYPE(__int16_t, a, b);
		case 4: return COMPARE_AS_TYPE(__int32_t, a, b);
		case 8: return COMPARE_AS_TYPE(__int64_t, a, b);
		case 16: return COMPARE_AS_TYPE(__int128_t, a, b);
	}
	return 0;
}

int atomic_data_validation_setup(MPI_Datatype datatype, int jrank, void *buf, size_t buf_size) {
	char set_value[64]; // fits maximum atom size of 256 bits.
	char *set_buf;
	int jatom;
	int dtype_size;
	size_t natoms;
	int err;
	
	set_buf = calloc(buf_size, 1);
	err = MPI_Type_size(datatype, &dtype_size);
	if (err) goto exit_path;

	natoms = buf_size/dtype_size;

	// get the value we wish to set the memory to.
	err = validation_input_value(datatype, jrank, set_value);
	if (err == -ENODATA) {
		err = 0;
		goto exit_path;
	}
	if (err) goto exit_path;



	// fill a system buffer with the value
	for (jatom=0; jatom < natoms; jatom++) {
		memcpy( set_buf + jatom*dtype_size, set_value, dtype_size );
	}

	// copy system buffer to hmem.
	err = set_hmem_buffer(buf, set_buf, buf_size );
exit_path:
	free(set_buf);
	return err;
}

#define PRINT_ADR_COMPARISON(dtype,fmt,ai,bi,ci,ao,ae) \
	fprintf(stderr, \
		"Initial Values: [local]addr=" fmt ", [remote]buf=" fmt ", [remote]compare=" fmt "\n" \
		"Observed Final Value: addr=" fmt "\n" \
		"Expected Final Value: addr=" fmt "\n", \
		*(ATOM_CTYPE_FOR_##dtype*)(ai), \
		*(ATOM_CTYPE_FOR_##dtype*)(bi), \
		*(ATOM_CTYPE_FOR_##dtype*)(ci), \
		*(ATOM_CTYPE_FOR_##dtype*)(ao), \
		*(ATOM_CTYPE_FOR_##dtype*)(ae) );
#define PRINT_ADR_COMPARISON_CPLX(fmtc,realfun,imagfun,ai,bi,ci,ao,ae) \
	fprintf(stderr, \
		"Initial Values: [local]addr=%"fmtc"%+"fmtc"i, [remote]buf=%"fmtc"%+"fmtc"i, [remote]compare=%"fmtc"%+"fmtc"i\n" \
		"Observed Final Value: addr=%"fmtc"%+"fmtc"i\n" \
		"Expected Final Value: addr=%"fmtc"%+"fmtc"i\n", \
		realfun(ai), imagfun(ai), \
		realfun(bi), imagfun(bi), \
		realfun(ci), imagfun(ci), \
		realfun(ao), imagfun(ao), \
		realfun(ae), imagfun(ae) );

#define PRINT_RES_COMPARISON(dtype,fmt,ai,bi,ci,ro,re) \
	fprintf(stderr, \
		"Initial Values: [remote]addr=" fmt ", [local]buf=" fmt ", [local]compare=" fmt "\n" \
		"Observed Final Value: result=" fmt "\n" \
		"Expected Final Value: result=" fmt "\n", \
		*(ATOM_CTYPE_FOR_##dtype*)(ai), \
		*(ATOM_CTYPE_FOR_##dtype*)(bi), \
		*(ATOM_CTYPE_FOR_##dtype*)(ci), \
		*(ATOM_CTYPE_FOR_##dtype*)(ro), \
		*(ATOM_CTYPE_FOR_##dtype*)(re) )
#define PRINT_RES_COMPARISON_CPLX(fmtc,realfun,imagfun,ai,bi,ci,ro,re) \
	fprintf(stderr, \
		"Initial Values: [remote]addr=%"fmtc"%+"fmtc"i, [local]buf=%"fmtc"%+"fmtc"i, [local]compare=%"fmtc"%+"fmtc"i\n" \
		"Observed Final Value: addr=%"fmtc"%+"fmtc"i\n" \
		"Expected Final Value: addr=%"fmtc"%+"fmtc"i\n", \
		realfun(ai), imagfun(ai), \
		realfun(bi), imagfun(bi), \
		realfun(ci), imagfun(ci), \
		realfun(ro), imagfun(ro), \
		realfun(re), imagfun(re) );


static void print_failure_message(MPI_Datatype datatype,
	void *adr_in, void *buf_in, void *compare_in,
	void *adr_obs, void *res_obs,
	void *adr_expect, void *res_expect)
{
	double complex dc;

	if (datatype == MPI_SIGNED_CHAR) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_SIGNED_CHAR,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_SIGNED_CHAR,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_UNSIGNED_CHAR) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_UNSIGNED_CHAR,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_UNSIGNED_CHAR,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_SHORT) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_SHORT,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_SHORT,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_UNSIGNED_SHORT) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_UNSIGNED_SHORT,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_UNSIGNED_SHORT,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_INT) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_INT,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_INT,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_UNSIGNED) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_UNSIGNED,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_UNSIGNED,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_LONG_LONG) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_LONG_LONG,"%ld",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_LONG_LONG,"%ld",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_UNSIGNED_LONG_LONG) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_UNSIGNED_LONG_LONG,"%lu",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_UNSIGNED_LONG_LONG,"%lu",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_LONG_DOUBLE) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_LONG_DOUBLE,"%Lf",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_LONG_DOUBLE,"%Lf",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_DOUBLE) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_DOUBLE,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_DOUBLE,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_LONG_DOUBLE) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_LONG_DOUBLE,"%Lf",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_LONG_DOUBLE,"%Lf",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_FLOAT) {
		if (adr_obs) PRINT_ADR_COMPARISON(DMPI_FLOAT,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
		if (res_obs) PRINT_RES_COMPARISON(DMPI_FLOAT,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
	}
	if (datatype == MPI_C_FLOAT_COMPLEX) {
		if (adr_obs) PRINT_ADR_COMPARISON_CPLX("f",crealf,cimagf,*(float complex*)adr_in,*(float complex*)buf_in,*(float complex*)compare_in,*(float complex*)adr_obs,*(float complex*)adr_expect);
		if (res_obs) PRINT_RES_COMPARISON_CPLX("f",crealf,cimagf,*(float complex*)adr_in,*(float complex*)buf_in,*(float complex*)compare_in,*(float complex*)res_obs,*(float complex*)res_expect);
	}
	if (datatype == MPI_C_DOUBLE_COMPLEX) {
		if (adr_obs) PRINT_ADR_COMPARISON_CPLX("f",creal,cimag,*(double complex*)adr_in,*(double complex*)buf_in,*(double complex*)compare_in,*(double complex*)adr_obs,*(double complex*)adr_expect);
		if (res_obs) PRINT_RES_COMPARISON_CPLX("f",creal,cimag,*(double complex*)adr_in,*(double complex*)buf_in,*(double complex*)compare_in,*(double complex*)res_obs,*(double complex*)res_expect);
	}
	if (datatype == MPI_C_LONG_DOUBLE_COMPLEX) {
		if (adr_obs) PRINT_ADR_COMPARISON_CPLX("Lf",creall,cimagl,*(long double complex*)adr_in,*(long double complex*)buf_in,*(long double complex*)compare_in,*(long double complex*)adr_obs,*(long double complex*)adr_expect);
		if (res_obs) PRINT_RES_COMPARISON_CPLX("Lf",creall,cimagl,*(long double complex*)adr_in,*(long double complex*)buf_in,*(long double complex*)compare_in,*(long double complex*)res_obs,*(long double complex*)res_expect);
	}
			// case FI_FLOAT:
			// 	if (adr_obs) PRINT_ADR_COMPARISON(FI_FLOAT,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
			// 	if (res_obs) PRINT_RES_COMPARISON(FI_FLOAT,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
			// 	break;
			// case FI_DOUBLE:
			// 	if (adr_obs) PRINT_ADR_COMPARISON(FI_DOUBLE,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
			// 	if (res_obs) PRINT_RES_COMPARISON(FI_DOUBLE,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
			// 	break;
			// default:
			// 	break;
}

int atomic_data_validation_check(MPI_Datatype datatype, MPI_Op op, int jrank, void *addr, void *res, size_t buf_size, bool check_addr, bool check_result) {
	// these all fit the maximum atom size of 256 bits.
	const int MAX_ATOM_BYTES=64;
	char local_addr[MAX_ATOM_BYTES],            remote_addr[MAX_ATOM_BYTES];
	char local_buf[MAX_ATOM_BYTES],             remote_buf[MAX_ATOM_BYTES];
	char local_compare[MAX_ATOM_BYTES],         remote_compare[MAX_ATOM_BYTES];
	char expected_local_addr[MAX_ATOM_BYTES],   dummy_remote_addr[MAX_ATOM_BYTES];
	char expected_local_result[MAX_ATOM_BYTES];

	char local_addr_in_sysmem[buf_size];
	char local_result_in_sysmem[buf_size];
	int dtype_size;
	size_t natoms;
	int jatom;
	int err, addr_eq, res_eq, any_errors=0;
	int jrank_remote = (jrank+1)%2;

 	err = MPI_Type_size(datatype, &dtype_size);
	if (err) return err;

	natoms = buf_size/dtype_size;

	// setup initial conditions so we can mock the test
	err  = validation_input_value(datatype, jrank, local_addr);
	err |= validation_input_value(datatype, jrank, local_buf);
	err |= validation_input_value(datatype, jrank, local_compare);
	err |= validation_input_value(datatype, jrank, expected_local_addr);
	err |= validation_input_value(datatype, jrank_remote, remote_addr);
	err |= validation_input_value(datatype, jrank_remote, remote_buf);
	err |= validation_input_value(datatype, jrank_remote, remote_compare);
	if (err == -ENODATA) goto nocheck;
	if (err) goto error;

	if ((long long)op == -1) {
		// mock the remote side performing CAS on our local addr
		err  = perform_atomic_cas(datatype, local_addr, remote_buf, expected_local_addr, remote_compare, NULL);
		// mock the local side performing CAS on remote addr
		err |= perform_atomic_cas(datatype, remote_addr, local_buf, dummy_remote_addr, local_compare, expected_local_result);
	}
	else {
		// mock the remote side performing operations on our local addr
		err  = perform_atomic_op(datatype, op, local_addr, remote_buf, expected_local_addr, remote_compare, NULL);
		// mock the local side performing operations on remote addr
		err |= perform_atomic_op(datatype, op, remote_addr, local_buf, dummy_remote_addr, local_compare, expected_local_result);
	}
	if (err == -ENODATA) goto nocheck;
	if (err) goto error;

	// if (datatype == MPI_C_LONG_DOUBLE_COMPLEX) {
	// 	printf("Checking: %Lf%+Lfi\t%Lf%+Lfi\t%Lf%+Lfi",
	// 		creall(local_addr),remote_addr, expected_local_addr);
	// }


	err  = get_hmem_buffer(local_addr_in_sysmem, addr, buf_size );
	err |= get_hmem_buffer(local_result_in_sysmem, res, buf_size );
	if (err) goto error;
	natoms = 1;
	for (jatom=0; jatom < natoms; jatom++) {
		addr_eq = 1;
		res_eq = 1;
		if (check_addr) {
			addr_eq = atom_binary_compare( datatype, expected_local_addr,
										   local_addr_in_sysmem + jatom*dtype_size);
		}
		if (!addr_eq) {
			fprintf( stderr, "FAILED: Remote atomic operation %s",fi_tostr(&op, FI_TYPE_ATOMIC_OP));
			fprintf(stderr, " on %s failed validation of addr at atom index %d.\n",
				fi_tostr(&datatype,    FI_TYPE_ATOMIC_TYPE),
				jatom );
			print_failure_message( datatype,
				local_addr, remote_buf, remote_compare,
				local_addr_in_sysmem + jatom*dtype_size, NULL,
				expected_local_addr, NULL);
		}
		if (check_result) {
			res_eq = atom_binary_compare( datatype, expected_local_result,
										  local_result_in_sysmem + jatom*dtype_size);
		}
		if (!res_eq) {
			fprintf( stderr, "FAILED: Local atomic operation %s",fi_tostr(&op, FI_TYPE_ATOMIC_OP));
			fprintf(stderr, " on %s failed validation of result at atom index %d.\n",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE),
				jatom );
			print_failure_message( datatype,
				remote_addr, local_buf, local_compare,
				NULL, local_result_in_sysmem + jatom*dtype_size,
				NULL, expected_local_result);
		}
		if (!res_eq || !addr_eq) {
			any_errors = 1;
			break;
		}
	}
	atomic_dv_record(datatype, op, any_errors, 1);
	return 0;

nocheck:
	atomic_dv_record(datatype, op, 0, 0);
	return 0;
error:
	atomic_dv_record(datatype, op, 0, 0);
	return err;


}

int is_mpi_cas_allowed(MPI_Datatype dtype) {
	if (dtype == MPI_C_FLOAT_COMPLEX)	return 0;
	if (dtype == MPI_C_DOUBLE_COMPLEX)	return 0;
	if (dtype == MPI_C_LONG_DOUBLE_COMPLEX)	return 0;
	return 1;
}

int is_mpi_op_allowed(MPI_Datatype dtype, MPI_Op op) {
	// see MPI standard v4.0 June 2021:  Sec 6.9.2, page 226
	// this function is not comprehensive, but it covers
	// most of the operations on C types that we intend to test.

	enum data_class { integer, floating_point, floating_complex};
	enum data_class dclass;

	if (dtype == MPI_DATATYPE_NULL)       return 0;
	if (dtype == MPI_SIGNED_CHAR)         dclass = integer;
	if (dtype == MPI_UNSIGNED_CHAR)       dclass = integer;
	if (dtype == MPI_SHORT)               dclass = integer;
	if (dtype == MPI_UNSIGNED_SHORT)      dclass = integer;
	if (dtype == MPI_INT)                 dclass = integer;
	if (dtype == MPI_UNSIGNED)            dclass = integer;
	if (dtype == MPI_LONG)                dclass = integer;
	if (dtype == MPI_UNSIGNED_LONG)       dclass = integer;
	if (dtype == MPI_LONG_LONG)           dclass = integer;
	if (dtype == MPI_UNSIGNED_LONG_LONG)  dclass = integer;
	if (dtype == MPI_FLOAT)               dclass = floating_point;
	if (dtype == MPI_DOUBLE)              dclass = floating_point;
	if (dtype == MPI_LONG_DOUBLE)         dclass = floating_point;
	if (dtype == MPI_C_FLOAT_COMPLEX)     dclass = floating_complex;
	if (dtype == MPI_C_DOUBLE_COMPLEX)    dclass = floating_complex;
	if (dtype == MPI_C_LONG_DOUBLE_COMPLEX) dclass = floating_complex;

	if (op == MPI_MAX || op == MPI_MIN)
		return dclass == integer || dclass == floating_point;
	if (op == MPI_SUM || op == MPI_PROD)
		return dclass == integer || dclass == floating_point || dclass == floating_complex;
	if (op == MPI_LAND || op == MPI_LOR || op == MPI_LXOR)
		return dclass == integer;
	if (op == MPI_BAND || op == MPI_BOR || op == MPI_BXOR)
		return dclass == integer;
	return 0;
}