#include <math.h>
#include "heydem.h"



// Load parameters for memory allocation
void LoadPara(char *DumpName, int *NP, int *nBatch, int *TotalParticle)
{
	char filename[20];
	strcpy(filename, DumpName);
	FILE *dpFile = fopen(filename, "rt");


	FindRec(dpFile,"SYSTEM Info");
	fscanf(dpFile, "%d", NP);
	fscanf(dpFile, "%d", nBatch);
	fscanf(dpFile, "%d", TotalParticle);
	fclose(dpFile);
}
