#include <math.h>
#include "heydem.h"



void UnitReduce(UNIT *Rdu, CylinderBC *h_Cir, FeedZone *Feedcube, LOADMODE *h_load, 
	            double *hRad, MatType *h_Mat, LIQUID *h_lqd, double InitParSize,  SIMULATION *h_dem, 
				PARAMETER *h_Params, SEARCHCELL h_cell, 
				double *h_Level_size, int nLevels,
				int NP, int RESTART, char* genfile)
{

	// if multiple materials exist, this snippet has to be modified
	Rdu->rlunit = InitParSize;                                           // length <diameter>
	Rdu->rmunit = h_Mat[0].density * PI * pow(InitParSize, 3.0) / 6.0;   // mass 
	Rdu->rfunit = Rdu->rmunit * gravity;								 // force
	Rdu->rsunit = Rdu->rfunit / pow(InitParSize, 2.0);                   // stress
	Rdu->rtunit = pow(InitParSize/gravity, 0.5);                         // time
	Rdu->rvunit = Rdu->rlunit / Rdu->rtunit;							// velocity
	Rdu->reunit = Rdu->rmunit * gravity * Rdu->rlunit;					// energy

	
	//*--- Boundary Condition ---*//
	h_Cir->cir.x = h_Cir->cir.x / Rdu->rlunit;
	h_Cir->cir.y = h_Cir->cir.y / Rdu->rlunit;
	h_Cir->Tw    = h_Cir->Tw    / Rdu->rlunit;
	h_Cir->Bw    = h_Cir->Bw    / Rdu->rlunit;
	h_Cir->R     = h_Cir->R     / Rdu->rlunit;


	// maximum particle size of each level
	for (int i=0; i<nLevels; i++)
	{
		h_Level_size[i] = h_Level_size[i] / Rdu->rlunit;
	}



	//*--- Feed Zone ---*//
	Feedcube->Lmax = Feedcube->Lmax / Rdu->rlunit;
	Feedcube->Lmin = Feedcube->Lmin / Rdu->rlunit;

	h_load->tPacking = h_load->tPacking / Rdu->rtunit;	

	

	//*----Particle Information----*//
	for(int k = 0; k<NP; k++)
	{
		hRad[k] = hRad[k] / Rdu->rlunit;
	}




	//*---Material Property----*//
	h_Mat[0].emod = h_Mat[0].emod * pow(Rdu->rlunit, 2.0) / Rdu->rfunit;  
	h_Mat[0].yldp = h_Mat[0].yldp * pow(Rdu->rlunit, 2.0) / Rdu->rfunit;
	h_Mat[0].dmpn = h_Mat[0].dmpn / Rdu->rtunit;

	h_Mat[1].emod = h_Mat[1].emod * pow(Rdu->rlunit, 2.0) / Rdu->rfunit;  
	h_Mat[1].yldp = h_Mat[1].yldp * pow(Rdu->rlunit, 2.0) / Rdu->rfunit;
	h_Mat[1].dmpn = h_Mat[1].dmpn / Rdu->rtunit;



	//*------------- capillary force -------------*//
	/*---------------------------------------------*/
	h_lqd->gapmn    = h_lqd->gapmn / InitParSize;
	h_lqd->layer    = 0.5 * ( pow((1.0 + h_lqd->vol), 1.0/3.0) - 1.0);  // Formula:(4/3)*PI*(R + L)**3 - (4/3)*PI*R**3 = Vol * (4/3)*PI*R**3
	h_lqd->vol      = h_lqd->vol * PI / 6.0;
	h_lqd->stension = h_lqd->stension * Rdu->rlunit / Rdu->rfunit;
	h_lqd->brknmx   = 0.99 * pow((h_lqd->vol / 6.0), 0.34);


	//*----reduced time step----*//
	h_dem->dt          = h_dem->dt / Rdu->rtunit;
	h_dem->outtime     = h_dem->outtime / Rdu->rtunit;
	h_dem->outinterval = h_dem->outinterval / Rdu->rtunit;
	h_dem->fdinterval  = h_dem->fdinterval / Rdu->rtunit;


   
	//*--- Parameters ---*//
	h_Params->Origin.x = h_Params->Origin.x / Rdu->rlunit;
	h_Params->Origin.y = h_Params->Origin.y / Rdu->rlunit;
	h_Params->Origin.z = h_Params->Origin.z / Rdu->rlunit;

	h_Params->EndPoint.x = h_Params->EndPoint.x / Rdu->rlunit;
	h_Params->EndPoint.y = h_Params->EndPoint.y / Rdu->rlunit;
	h_Params->EndPoint.z = h_Params->EndPoint.z / Rdu->rlunit;


	for (int i=0; i<nLevels; i++)
	{
		h_cell.Cellsize[i] = h_cell.Cellsize[i] / Rdu->rlunit;
	}
	


	// copy dt to constant variable
	h_Params->dt = h_dem->dt;  
	h_Params->NP = NP; 
	h_Params->SearchGap = h_dem->cutGap; 




	FILE *LogFile; 
	char LogjName[20];
	strcpy(LogjName, genfile);
	strcat(LogjName ,"_Log.dat"); 

	char *mode = "a";
	LogFile = fopen(LogjName, mode);

	fprintf(LogFile, "\n\n==================================\n");
	fprintf(LogFile, "<*--- Dimensionless Variable ---*>\n");
	fprintf(LogFile, "=======================================================\n");

	fprintf(LogFile, "<size>: %lf mm \t\t", Rdu->rlunit * 1.0e3);
	fprintf(LogFile, "<time>: %lf \n", Rdu->rtunit);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<mass>: %lf e-6\t\t", Rdu->rmunit*1.0e6);
	fprintf(LogFile, "<force>: %lf e-6\n", Rdu->rfunit*1.0e6);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<stress>: %lf e-6\t\t", Rdu->rsunit*1.0e6);
	fprintf(LogFile, "<velocity>: %lf \n", Rdu->rvunit);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<energy>: %lf \n", Rdu->reunit);
	fprintf(LogFile, "=======================================================\n\n\n");
	fclose(LogFile);


}
