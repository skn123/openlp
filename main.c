/*
 * Author:		Chris Wailes <chris.wailes@gmail.com> and
 * 				Jonathan Turner <jonathan.turner@colorado.edu>
 * Project:		CS 5654 PA1
 * Date:			2011/10/16
 * Description:	The main driver of our program.
 */

// Standard Incldues
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Project Includes
#include "dictionary.h"
#include "input.h"
#include "output.h"
#include "util.h"

// Global Variables

config_t cfg;

// Functions

int main(int argc, char** argv) {
	dict_t* dict;
	uint num_orig_vars;

	dict = load_lp_file();
	
	if (cfg.mathprog_filename) {
		output_glpsol(dict, cfg.mathprog_filename);
		exit(0);
	}

	if (cfg.verbose) {
		printf("Initial Dictionary:\n");
		dict_view(dict);
	}

	// Initialize the dictionary proper.
	if (dict_init(dict) && cfg.verbose) {
		printf("Dictionary after initialization:\n");
		dict_view(dict);
	}
	
	// TODO: do simplex here
	
	if (cfg.verbose) {
		printf("Final Dictionary:\n");
		dict_view(dict);
	}
	
	printf("Final Values:\n");
	dict_view_answer(dict, num_orig_vars);
	printf("\n");
	
	dict_free(dict);
	return 0;
}
