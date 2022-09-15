#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************/
/*                                                             */
/* Files: isaprogram   LC-3 machine language program file     */
/*                                                             */
/***************************************************************/

/***************************************************************/
/* These are the functions you'll have to write.               */
/***************************************************************/

void process_instruction();

/***************************************************************/
/* A couple of useful definitions.                             */
/***************************************************************/
#define FALSE 0
#define TRUE  1

/***************************************************************/
/* Use this to avoid overflowing 16 bits on the bus.           */
/***************************************************************/
#define Low16bits(x) ((x) & 0xFFFF)

/***************************************************************/
/* Main memory.                                                */
/***************************************************************/
/* 
  MEMORY[A] stores the word address A
*/

#define WORDS_IN_MEM    0x08000 
int MEMORY[WORDS_IN_MEM];

/***************************************************************/

/***************************************************************/

/***************************************************************/
/* LC-3 State info.                                           */
/***************************************************************/
#define LC_3_REGS 8

int RUN_BIT;	/* run bit */


typedef struct System_Latches_Struct{

  int PC,		/* program counter */
    N,		/* n condition bit */
    Z,		/* z condition bit */
    P;		/* p condition bit */
  int REGS[LC_3_REGS]; /* register file. */
} System_Latches;

/* Data Structure for Latch */

System_Latches CURRENT_LATCHES, NEXT_LATCHES;

/***************************************************************/
/* A cycle counter.                                            */
/***************************************************************/
int INSTRUCTION_COUNT;

/***************************************************************/
/*                                                             */
/* Procedure : help                                            */
/*                                                             */
/* Purpose   : Print out a list of commands                    */
/*                                                             */
/***************************************************************/
void help() {                                                    
  printf("----------------LC-3 ISIM Help-----------------------\n");
  printf("go               -  run program to completion         \n");
  printf("run n            -  execute program for n instructions\n");
  printf("mdump low high   -  dump memory from low to high      \n");
  printf("rdump            -  dump the register & bus values    \n");
  printf("?                -  display this help menu            \n");
  printf("quit             -  exit the program                  \n\n");
}

/***************************************************************/
/*                                                             */
/* Procedure : cycle                                           */
/*                                                             */
/* Purpose   : Execute a cycle                                 */
/*                                                             */
/***************************************************************/
void cycle() {                                                

  process_instruction();
  CURRENT_LATCHES = NEXT_LATCHES;
  INSTRUCTION_COUNT++;
}

/***************************************************************/
/*                                                             */
/* Procedure : run n                                           */
/*                                                             */
/* Purpose   : Simulate the LC-3 for n cycles                 */
/*                                                             */
/***************************************************************/
void run(int num_cycles) {                                      
  int i;

  if (RUN_BIT == FALSE) {
    printf("Can't simulate, Simulator is halted\n\n");
    return;
  }

  printf("Simulating for %d cycles...\n\n", num_cycles);
  for (i = 0; i < num_cycles; i++) {
    if (CURRENT_LATCHES.PC == 0x0000) {
	    RUN_BIT = FALSE;
	    printf("Simulator halted\n\n");
	    break;
    }
    cycle();
  }
}

/***************************************************************/
/*                                                             */
/* Procedure : go                                              */
/*                                                             */
/* Purpose   : Simulate the LC-3 until HALTed                 */
/*                                                             */
/***************************************************************/
void go() {                                                     
  if (RUN_BIT == FALSE) {
    printf("Can't simulate, Simulator is halted\n\n");
    return;
  }

  printf("Simulating...\n\n");
  while (CURRENT_LATCHES.PC != 0x0000)
    cycle();
  RUN_BIT = FALSE;
  printf("Simulator halted\n\n");
}

/***************************************************************/ 
/*                                                             */
/* Procedure : mdump                                           */
/*                                                             */
/* Purpose   : Dump a word-aligned region of memory to the     */
/*             output file.                                    */
/*                                                             */
/***************************************************************/
void mdump(FILE * dumpsim_file, int start, int stop) {          
  int address; /* this is a address */

  printf("\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
  printf("-------------------------------------\n");
  for (address = start ; address <= stop ; address++)
    printf("  0x%.4x (%d) : 0x%.2x\n", address , address , MEMORY[address]);
  printf("\n");

  /* dump the memory contents into the dumpsim file */
  fprintf(dumpsim_file, "\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
  fprintf(dumpsim_file, "-------------------------------------\n");
  for (address = start ; address <= stop ; address++)
    fprintf(dumpsim_file, " 0x%.4x (%d) : 0x%.2x\n", address , address , MEMORY[address]);
  fprintf(dumpsim_file, "\n");
  fflush(dumpsim_file);
}

/***************************************************************/
/*                                                             */
/* Procedure : rdump                                           */
/*                                                             */
/* Purpose   : Dump current register and bus values to the     */   
/*             output file.                                    */
/*                                                             */
/***************************************************************/
void rdump(FILE * dumpsim_file) {                               
  int k; 

  printf("\nCurrent register/bus values :\n");
  printf("-------------------------------------\n");
  printf("Instruction Count : %d\n", INSTRUCTION_COUNT);
  printf("PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
  printf("CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
  printf("Registers:\n");
  for (k = 0; k < LC_3_REGS; k++)
    printf("%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
  printf("\n");

  /* dump the state information into the dumpsim file */
  fprintf(dumpsim_file, "\nCurrent register/bus values :\n");
  fprintf(dumpsim_file, "-------------------------------------\n");
  fprintf(dumpsim_file, "Instruction Count : %d\n", INSTRUCTION_COUNT);
  fprintf(dumpsim_file, "PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
  fprintf(dumpsim_file, "CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
  fprintf(dumpsim_file, "Registers:\n");
  for (k = 0; k < LC_3_REGS; k++)
    fprintf(dumpsim_file, "%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
  fprintf(dumpsim_file, "\n");
  fflush(dumpsim_file);
}

/***************************************************************/
/*                                                             */
/* Procedure : get_command                                     */
/*                                                             */
/* Purpose   : Read a command from standard input.             */  
/*                                                             */
/***************************************************************/
void get_command(FILE * dumpsim_file) {                         
  char buffer[20];
  int start, stop, cycles;

  printf("LC-3-SIM> ");

  scanf("%s", buffer);
  printf("\n");

  switch(buffer[0]) {
  case 'G':
  case 'g':
    go();
    break;

  case 'M':
  case 'm':
    scanf("%i %i", &start, &stop);
    mdump(dumpsim_file, start, stop);
    break;

  case '?':
    help();
    break;
  case 'Q':
  case 'q':
    printf("Bye.\n");
    exit(0);

  case 'R':
  case 'r':
    if (buffer[1] == 'd' || buffer[1] == 'D')
	    rdump(dumpsim_file);
    else {
	    scanf("%d", &cycles);
	    run(cycles);
    }
    break;

  default:
    printf("Invalid Command\n");
    break;
  }
}

/***************************************************************/
/*                                                             */
/* Procedure : init_memory                                     */
/*                                                             */
/* Purpose   : Zero out the memory array                       */
/*                                                             */
/***************************************************************/
void init_memory() {                                           
  int i;

  for (i=0; i < WORDS_IN_MEM; i++) {
    MEMORY[i] = 0;
  }
}

/**************************************************************/
/*                                                            */
/* Procedure : load_program                                   */
/*                                                            */
/* Purpose   : Load program and service routines into mem.    */
/*                                                            */
/**************************************************************/
void load_program(char *program_filename) {                   
  FILE * prog;
  int ii, word, program_base;

  /* Open program file. */
  prog = fopen(program_filename, "r");
  if (prog == NULL) {
    printf("Error: Can't open program file %s\n", program_filename);
    exit(-1);
  }

  /* Read in the program. */
  if (fscanf(prog, "%x\n", &word) != EOF)
    program_base = word ;
  else {
    printf("Error: Program file is empty\n");
    exit(-1);
  }

  ii = 0;
  while (fscanf(prog, "%x\n", &word) != EOF) {
    /* Make sure it fits. */
    if (program_base + ii >= WORDS_IN_MEM) {
	    printf("Error: Program file %s is too long to fit in memory. %x\n",
             program_filename, ii);
	    exit(-1);
    }

    /* Write the word to memory array. */
    MEMORY[program_base + ii] = word;
    ii++;
  }

  if (CURRENT_LATCHES.PC == 0) CURRENT_LATCHES.PC = program_base;

  printf("Read %d words from program into memory.\n\n", ii);
}

/************************************************************/
/*                                                          */
/* Procedure : initialize                                   */
/*                                                          */
/* Purpose   : Load machine language program                */ 
/*             and set up initial state of the machine.     */
/*                                                          */
/************************************************************/
void initialize(char *program_filename, int num_prog_files) { 
  int i;

  init_memory();
  for ( i = 0; i < num_prog_files; i++ ) {
    load_program(program_filename);
    while(*program_filename++ != '\0');
  }
  CURRENT_LATCHES.Z = 1;  
  NEXT_LATCHES = CURRENT_LATCHES;
    
  RUN_BIT = TRUE;
}

/***************************************************************/
/*                                                             */
/* Procedure : main                                            */
/*                                                             */
/***************************************************************/
int main(int argc, char *argv[]) {                              
  FILE * dumpsim_file;

  /* Error Checking */
  if (argc < 2) {
    printf("Error: usage: %s <program_file_1> <program_file_2> ...\n",
           argv[0]);
    exit(1);
  }

  printf("LC-3 Simulator\n\n");

  initialize(argv[1], argc - 1);

  if ( (dumpsim_file = fopen( "dumpsim", "w" )) == NULL ) {
    printf("Error: Can't open dumpsim file\n");
    exit(-1);
  }

  while (1)
    get_command(dumpsim_file);
    
}

/***************************************************************/
/* Do not modify the above code.
   You are allowed to use the following global variables in your
   code. These are defined above.

   MEMORY

   CURRENT_LATCHES
   NEXT_LATCHES

   You may define your own local/global variables and functions.
   You may use the functions to get at the control bits defined
   above.

   Begin your code here 	  			       */

/***************************************************************/


/* ====================================================================
 *           Some Constantly Used Functions and Subroutines
 * ==================================================================== */

int IR_BIN[16];                          // binary representation of Instruction
void IST_toBin(int instruction)
{
  int x = instruction, bit = 0;
  while (x)
  {
    IR_BIN[bit] = x % 2;
    x = x >> 1;
    bit++;
  }
  for (int i=bit; i<16; i++)
    IR_BIN[i] = 0;
}

void set_cc(int result)
{
  NEXT_LATCHES.N = (result > 32767);
  NEXT_LATCHES.P = (result > 0 && result <= 32767);
  NEXT_LATCHES.Z = (result == 0);
}

int sext(int bits)                        // SEXT for 2's complement. 
{
  int offset = 0;
  for (int i=15; i>=bits-1; i--)
    offset = (offset<<1) + IR_BIN[bits-1];
  for (int i=bits-2; i>=0; i--)
    offset = (offset<<1) + IR_BIN[i];
  return offset;
}

int sext_offset(int bits)                  // the SEXT value. 
{
  int offset = -IR_BIN[bits-1];
  for (int i=bits-2; i>=0; i--)
    offset = (offset<<1) + IR_BIN[i];
  return offset;
}

/* ====================================================================
 *
 *   Part of the Decode Phase and the Execution Phase of Instructions
 *
 * ====================================================================
 *
 * Specific functions for each type of instruction is given below.   
 */
 
void add_(int dst, int sr1)
{
  int result;
  if (IR_BIN[5] == 0)
  {
    int sr2 = (IR_BIN[2]<<2) + (IR_BIN[1]<<1) + IR_BIN[0];
    result = Low16bits(CURRENT_LATCHES.REGS[sr1] + CURRENT_LATCHES.REGS[sr2]);
  }
  else
  {
    int imm = sext(5);
    result = Low16bits(CURRENT_LATCHES.REGS[sr1] + imm);
  }
  NEXT_LATCHES.REGS[dst] = result; 
  set_cc(result); 
}

void and_(int dst, int sr1)
{
  int result;
  if (IR_BIN[5] == 0)
  {
    int sr2 = (IR_BIN[2]<<2) + (IR_BIN[1]<<1) + IR_BIN[0];
    result = Low16bits(CURRENT_LATCHES.REGS[sr1] & CURRENT_LATCHES.REGS[sr2]);
  }
  else
  {
    int imm = sext(5);
    result = Low16bits(CURRENT_LATCHES.REGS[sr1] & imm);
  }
  NEXT_LATCHES.REGS[dst] = result; 
  set_cc(result); 
}

void not_(int dst, int src)
{
  NEXT_LATCHES.REGS[dst] = Low16bits(~CURRENT_LATCHES.REGS[src]);
  set_cc(NEXT_LATCHES.REGS[dst]);
}

void branch_() 
{
  if (!((IR_BIN[11] && CURRENT_LATCHES.N) || (IR_BIN[10] && CURRENT_LATCHES.Z) ||
       (IR_BIN[9] && CURRENT_LATCHES.P)))
    return;
  NEXT_LATCHES.PC = CURRENT_LATCHES.PC + sext_offset(9);
}

void jmp_(int baseR)
{
  NEXT_LATCHES.PC = CURRENT_LATCHES.REGS[baseR];
}

void jsr_(int baseR)
{
  if (IR_BIN[11] == 0)
  {
    NEXT_LATCHES.PC = CURRENT_LATCHES.REGS[baseR];
  }
  else
  {
    NEXT_LATCHES.PC = CURRENT_LATCHES.PC + sext_offset(9);
  }
  NEXT_LATCHES.REGS[7] = CURRENT_LATCHES.PC;
}

void ld_(int dst)
{
  NEXT_LATCHES.REGS[dst] = MEMORY[ CURRENT_LATCHES.PC + sext_offset(9) ];
  set_cc(NEXT_LATCHES.REGS[dst]);
}

void ldi_(int dst)
{
  NEXT_LATCHES.REGS[dst] = MEMORY[ MEMORY[CURRENT_LATCHES.PC+sext_offset(9)] ];
  set_cc(NEXT_LATCHES.REGS[dst]);
}

void ldr_(int dst, int baseR)
{
  NEXT_LATCHES.REGS[dst] = MEMORY[ CURRENT_LATCHES.REGS[baseR]+sext_offset(6) ];
  set_cc(NEXT_LATCHES.REGS[dst]);
}

void lea_(int dst)
{
  NEXT_LATCHES.REGS[dst] = CURRENT_LATCHES.PC + sext_offset(9);
}

void st_(int src)
{
  MEMORY[ CURRENT_LATCHES.PC + sext_offset(9) ] = CURRENT_LATCHES.REGS[src];
}

void sti_(int src)
{
  MEMORY[ MEMORY[CURRENT_LATCHES.PC+sext_offset(9)] ] = CURRENT_LATCHES.REGS[src];
}

void str_(int src, int baseR)
{
  MEMORY[ CURRENT_LATCHES.REGS[baseR] + sext_offset(6) ] = CURRENT_LATCHES.REGS[src];
}

void trap_()
{
  IR_BIN[8] = 0;
  int zext_offset = sext_offset(9);   // since IR_BIN[8] is assigned 0, sext_offset(9) returns ZEXT(vec8).
  NEXT_LATCHES.PC = MEMORY[zext_offset];
}

/* ====================================================================
 *
 *   The Main Function of Executing Instructions at Each Cycle
 *
 * ==================================================================== */

void process_instruction(){
  /*  function: process_instruction
   *  
   *    Process one instruction at a time  
   *       -Fetch one instruction
   *       -Decode 
   *       -Execute
   *       -Update NEXT_LATCHES
   */
  
  int IR = MEMORY[CURRENT_LATCHES.PC];
  IST_toBin(IR);
  CURRENT_LATCHES.PC++;

  int IST_CODE = (IR_BIN[15]<<3) + (IR_BIN[14]<<2) + (IR_BIN[13]<<1) + IR_BIN[12];    // instruction code
  int dst = (IR_BIN[11]<<2) + (IR_BIN[10]<<1) + IR_BIN[9];
  int src = (IR_BIN[8]<<2) + (IR_BIN[7]<<1) + IR_BIN[6];

  NEXT_LATCHES = CURRENT_LATCHES;
  switch (IST_CODE)
  {
    case 1:  add_(dst, src);   break;
    case 5:  and_(dst, src);   break;
    case 0:  branch_();        break;
    case 12: jmp_(src);        break;
    case 4:  jsr_(src);        break;
    case 2:  ld_(dst);         break;
    case 10: ldi_(dst);        break;
    case 6:  ldr_(dst, src);   break;
    case 14: lea_(dst);        break;
    case 9:  not_(dst, src);   break;
    case 3:  st_(dst);         break;
    case 11: sti_(dst);        break;
    case 7:  str_(dst, src);   break;
    case 15: trap_();          break;
  }

}
