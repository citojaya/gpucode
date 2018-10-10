#include <math.h>
#include "heydem.h"
#include "double_math.h"




void DiaInput(char *diaFile, double *hRad, int NP)
{

	char filename[20];
	strcpy(filename, diaFile);
	FILE *PdiaFile = fopen(filename, "rt");


	int Tnp = NP;
	double *diaTemp = new double[Tnp];
	int num = 0;


	FindRec(PdiaFile, "PARTICLE DIAMETER");
	while (!feof(PdiaFile))
	{
		fscanf(PdiaFile, "%lf", &diaTemp[num]);
		printf("D[%d]: %lf \n", num, diaTemp[num]);
		num = num + 1;
	}



	printf("num of diameters: %d. \n", num);
	int k = 0;
	for (int i=0; i<Tnp/num+1; i++)
	{
		for (int j=0; j<num; j++)
		{
			if (k < Tnp)
			{
				// radius
				hRad[k] = diaTemp[j] / 2.0f; 
				k = k + 1;
			}
		}
	}

	
	delete[] diaTemp;
	fclose(PdiaFile);
}



