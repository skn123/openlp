/*
 * Author:		Chris Wailes <chris.wailes@gmail.com> and
 * 				Jonathan Turner <jonathan.turner@colorado.edu>
 * Project:		CS 5654 PA1
 * Date:			2011/10/16
 * Description:	Functions for manipulating dictionaries.
 */

// Standard Incldues
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Project Includes
#include "dictionary.h"
#include "matrix.h"
#include "util.h"

// Globals

extern config_t cfg;

// Functions

void dict_free(dict_t* dict) {
	free(dict->objective);
	free(dict->row_values);
	
	matrix_free(&dict->matrix);
	
	free(dict->col_labels);
	free(dict->row_labels);
	
	free(dict->row_bounds.upper);
	free(dict->row_bounds.lower);
	free(dict->col_bounds.upper);
	free(dict->col_bounds.lower);
	
	free(dict->col_rests);
	
	free(dict);
}

/*
 * FIXME:	This could possibly be made faster by obtaining a reference to the
 * 		relvent row and then indexing into it, as opposed to re-calculating
 * 		the row pointer each time.  However, I do believe that inlining and
 * 		common sub-expression elimination should take care of that for us.
 */
double dict_get_constraint_value(const dict_t* dict, uint con_index) {
	uint col_index;
	double con_val = 0;
	
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		con_val += matrix_get_value(&dict->matrix, con_index, col_index) * dict_get_var_bound_value(dict, col_index);
	}
	
	return con_val;
}

iset_t dict_get_infeasible_rows(const dict_t* dict) {
	uint row_index;
	double con_val;
	
	iset_t iset;
	
	iset.rows		= NULL;
	iset.num_rows	= 0;

	for (row_index = 0; row_index < dict->num_cons; ++row_index) {
		con_val = dict->row_values[row_index];
		
		if ((con_val < dict->row_bounds.lower[row_index]) || (con_val > dict->row_bounds.upper[row_index])) {

			// Increment the number of infeasible rows.
			++iset.num_rows;
			
			// Allocate enough space for the new irow_t element.
			iset.rows = realloc(iset.rows, iset.num_rows * sizeof(irow_t));
			
			iset.rows[iset.num_rows - 1].row_index	= row_index;
			iset.rows[iset.num_rows - 1].amount	= (con_val < dict->row_bounds.lower[row_index] ? dict->row_bounds.lower : dict->row_bounds.upper)[row_index] - con_val;
		}
	}

	return iset;
}

uint dict_get_num_unbounded_vars(const dict_t* dict) {
	uint col_index;
	uint num_unbounded_vars = 0;
	
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		if (dict_var_is_unbounded(dict, col_index)) {
			++num_unbounded_vars;
		}
	}
	
	return num_unbounded_vars;
}

inline double dict_get_var_bound_value(const dict_t* dict, uint var_index) {
	return (dict->col_rests[var_index] == UPPER ? dict->col_bounds.upper : dict->col_bounds.lower)[var_index];
}

double dict_get_var_value_by_label(const dict_t* dict, uint var_label) {
	uint col_index, row_index;
	double var_total;
	
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		if (dict->col_labels[col_index] == var_label) {
			
			var_total = dict_get_var_bound_value(dict, col_index);

			if (dict->split_vars[var_label]) {
				var_total -= dict_get_var_value_by_label(dict, dict->split_vars[var_label]);
			}

			return var_total;
		}
	}

	for (row_index = 0; row_index < dict->num_cons; ++row_index) {
		if (dict->row_labels[row_index] == var_label) {

			var_total = dict->row_values[row_index];

			if (dict->split_vars[var_label]) {
				var_total -= dict_get_var_value_by_label(dict, dict->split_vars[var_label]);
			}

			return var_total;
		}
	}
	
	fprintf(stderr, "Unknown variable request: x%u\n", var_label);
	exit(-1);
}

bool dict_init(dict_t* dict) {
        // TODO: Add dictionary init code here
	return TRUE;
}

bool dict_is_final(const dict_t* dict) {
	uint col_index;
	
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		if ((dict->objective[col_index] < 0 && dict->col_rests[col_index] == UPPER) || (dict->objective[col_index] > 0 && dict->col_rests[col_index] == LOWER)) {
			return FALSE;
		}
	}
	
	return TRUE;
}

dict_t* dict_new(uint num_vars, uint num_cons) {
	dict_t* dict;
	
	/*
	 * Allocate the necessary memory.
	 */
	
	dict = malloc(sizeof(dict_t));
	
	// Initialize the matrix.
	matrix_init(&dict->matrix, num_cons, num_vars);
	
	dict->objective	= (double*)malloc(num_vars * sizeof(double));
	dict->row_values	= (double*)malloc(num_cons * sizeof(double));
	
	dict->col_labels = (uint*)malloc(num_vars * sizeof(uint));
	dict->row_labels = (uint*)malloc(num_cons * sizeof(uint));
	
	dict->row_bounds.upper = (double*)malloc(num_cons * sizeof(double));
	dict->row_bounds.lower = (double*)malloc(num_cons * sizeof(double));
	
	dict->col_bounds.upper = (double*)malloc(num_vars * sizeof(double));
	dict->col_bounds.lower = (double*)malloc(num_vars * sizeof(double));
	
	dict->col_rests = (rest_t*)malloc(num_vars * sizeof(rest_t));
	
	dict->split_vars = (uint*)malloc((num_vars+1) * sizeof(uint));
	memset(dict->split_vars, 0, (num_vars+1) * sizeof(uint));
	
	// Set the number of variables and constraints for the dictionary.
	dict->num_vars = num_vars;
	dict->num_cons = num_cons;
	
	return dict;
}

void dict_resize(dict_t* dict, uint new_num_vars, uint new_num_cons) {
	//In case we want to snapshot the previous pointers, don't realloc them.
	//Instead, just create a new dictionary and replace the pointers.
	
	uint* new_row_labels;
	uint* new_col_labels;
	uint* new_split_vars;
	double* new_objective;
	
	rest_t* new_var_rests;
	bounds_t new_var_bounds, new_con_bounds;
	
	if (new_num_vars == dict->num_vars && new_num_cons == dict->num_cons) {
		return;
	}
	
	new_objective = malloc(sizeof(double) * new_num_vars);
	memset(new_objective, 0, sizeof(double) * new_num_vars);
	memcpy(new_objective, dict->objective, sizeof(double) * MIN(new_num_vars, dict->num_vars));
	dict->objective = new_objective;

	new_col_labels = malloc(sizeof(uint) * new_num_vars);
	memset(new_col_labels, 0, sizeof(uint) * new_num_vars);
	memcpy(new_col_labels, dict->col_labels, sizeof(uint) * MIN(new_num_vars, dict->num_vars));
	dict->col_labels = new_col_labels;

	new_var_rests = malloc(sizeof(rest_t) * new_num_vars);
	memset(new_var_rests, 0, sizeof(rest_t) * new_num_vars);
	memcpy(new_var_rests, dict->col_rests, sizeof(rest_t) * MIN(new_num_vars, dict->num_vars));
	dict->col_rests = new_var_rests;

	new_split_vars = malloc(sizeof(uint) * (new_num_vars + 1));
	memset(new_split_vars, 0, sizeof(uint) * (new_num_vars + 1));
	memcpy(new_split_vars, dict->split_vars, sizeof(uint) * (MIN(new_num_vars, dict->num_vars) + 1));
	dict->split_vars = new_split_vars;
	
	new_var_bounds.lower = malloc(sizeof(double) * new_num_vars);
	memset(new_var_bounds.lower, 0, (sizeof(double) * new_num_vars));

	new_var_bounds.upper = malloc(sizeof(double) * new_num_vars);
	memset(new_var_bounds.upper, 0, (sizeof(double) * new_num_vars));

	memcpy(new_var_bounds.lower, dict->col_bounds.lower, sizeof(double) * MIN(new_num_vars, dict->num_vars));
	memcpy(new_var_bounds.upper, dict->col_bounds.upper, sizeof(double) * MIN(new_num_vars, dict->num_vars));
	dict->col_bounds = new_var_bounds;

	new_row_labels = malloc(sizeof(uint) * new_num_cons);
	memset(new_row_labels, 0, (sizeof(uint) * new_num_cons));
	memcpy(new_row_labels, dict->row_labels, sizeof(uint) * MIN(new_num_cons, dict->num_cons));
	dict->row_labels = new_row_labels;

	new_con_bounds.lower = malloc(sizeof(double) * new_num_cons);
	memset(new_con_bounds.lower, 0, (sizeof(double) * new_num_cons));

	new_con_bounds.upper = malloc(sizeof(double) * new_num_cons);
	memset(new_con_bounds.upper, 0, (sizeof(double) * new_num_cons));

	memcpy(new_con_bounds.lower, dict->row_bounds.lower, sizeof(double) * MIN(new_num_cons, dict->num_cons));
	memcpy(new_con_bounds.upper, dict->row_bounds.upper, sizeof(double) * MIN(new_num_cons, dict->num_cons));
	dict->row_bounds = new_con_bounds;
	
	dict->row_values = realloc(dict->row_values, new_num_cons * sizeof(double));
	
	matrix_resize(&dict->matrix, new_num_cons, new_num_vars);

	dict->num_cons = new_num_cons;
	dict->num_vars = new_num_vars;
}

void dict_set_bounds_and_values(dict_t* dict) {
	uint col_index, row_index;
	
	// Pick the initial resting bounds for the variables.
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		if ((dict->objective[col_index] >= 0 && dict->col_bounds.upper[col_index] != INFINITY) || (dict->col_bounds.lower[col_index] == -INFINITY)) {
			dict->col_rests[col_index] = UPPER;
			
		} else {
			dict->col_rests[col_index] = LOWER;
		}
	}
	
	// Calculate the initial values of the constraints.
	for (row_index = 0; row_index < dict->num_cons; ++row_index) {
		dict->row_values[row_index] = dict_get_constraint_value(dict, row_index);
	}
}

inline bool dict_var_is_unbounded(const dict_t* dict, uint var_index) {
	return dict->col_bounds.upper[var_index] == INFINITY && dict->col_bounds.lower[var_index] == -INFINITY;
}

void dict_view(const dict_t* dict) {
	uint col_index, row_index;
	char buffer[10];
	
	// Print column labels.
	printf("                        ");
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		snprintf(buffer, 10, "x%u", dict->col_labels[col_index]);
		printf("%8s", buffer);
	}
	printf("     value\n");
	
	// Print bounds, labels, and values for rows.
	for (row_index = 0; row_index < dict->num_cons; ++row_index) {
		// Format the column label.
		snprintf(buffer, 10, "x%u", dict->row_labels[row_index]);
		
		// Print out bounds and label info.
		printf("% 7.3g % 7.3g | %4s |", dict->row_bounds.lower[row_index], dict->row_bounds.upper[row_index], buffer);
		
		for (col_index = 0; col_index < dict->num_vars; ++col_index) {
			printf(" % 7.3g", matrix_get_value(&dict->matrix, row_index, col_index));
		}
		
		// Print the constraint's value.
		printf(" | % 7.3g", dict->row_values[row_index]);
		
		printf("\n");
	}
	
	// Print seperator.
	printf("----------------------------------");
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		printf("--------");
	}
	printf("\n");
	
	// Print objective function coefficients.
	printf("                     z |");
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		printf(" % 7.3g", dict->objective[col_index]);
	}
	printf("\n");
	
	// Print the variables' lower bounds.
	printf("                       | ");
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		printf(dict->col_rests[col_index] == LOWER ? " [% 5.2g]" : "  % 5.2g ", dict->col_bounds.lower[col_index]);
	}
	printf("\n");
	
	// Print the variables' upper bounds.
	printf("                       | ");
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		printf(dict->col_rests[col_index] == UPPER ? " [% 5.2g]" : "  % 5.2g ", dict->col_bounds.upper[col_index]);
	}
	printf("\n\n");
}

void dict_view_answer(const dict_t* dict, uint num_orig_vars) {
	uint col_index, var_index;
	double objective = 0.0;
	char buffer[10];
	
	for (col_index = 0; col_index < dict->num_vars; ++col_index) {
		objective += dict->objective[col_index] * dict_get_var_bound_value(dict, col_index);
	}

	printf("\t   z = %- 7.3g\n", objective);

	for (var_index = 1; var_index <= num_orig_vars; ++var_index) {
		snprintf(buffer, 10, "x%u", var_index);
		
		printf("\t%4s = %- 7.3g\n", buffer, dict_get_var_value_by_label(dict, var_index));
	}
}
