#include <math.h>
#include "heydem.h"



//*--- Find the reading position in the file--*/
void FindRec(FILE *inFile, char* strDest)
{
	int nbytes = 256;
	char* strSrc;
	strSrc = (char *)malloc(nbytes+1);

	rewind(inFile);
	int n=strlen(strDest);
	while(!feof(inFile))
	{
		fgets(strSrc, 256, inFile);
		strSrc[n]='\0';
		if (strcmp(strDest, strSrc) == 0)
		{
			break;
		}
	}

	if(strcmp(strDest, strSrc) != 0)
	{
		//free(strSrc);
		//printf("Unable to find relevant info of: %s \n", strDest);
		exit(1);
	}
	free(strSrc);
}







//*---read from file or set by user----*//
void ReadData(double *h_Level_size, int *h_pLevel, int *h_level_nNebors, int *h_level_nPars,int *h_Nebor_Idx,
	          CylinderBC *&h_Cir,   CylinderBC *&d_Cir, FeedZone *Feedcube,
			  MatType *h_Mat, LIQUID *h_lqd, PARAMETER *h_Params, double *InitParSize, 
			  int* ForceCap,  SIMULATION *h_dem, LOADMODE *h_load, 
			  double *hRad, SEARCHCELL h_cell, int NP, int nLevels, char* genfile, int RESTART)
{
	// input file reading
	char filename[20];
	strcpy(filename, genfile);
	strcat(filename ,".in"); 
	FILE *InFile = fopen(filename, "rt");
	

	if (InFile == NULL)
	{
		fprintf(stderr, "Can't open the parameter file %s! \n", filename);
		char c = getchar();
		exit(1);
	}

	char LogjName[20];
	strcpy(LogjName, genfile);
	strcat(LogjName ,"_Log.dat"); 
	FILE *LogFile = fopen(LogjName, "a");


	// expand size for computing area
	double exComp = 0.0;

	// loading phase
	double DieDepth = 1.0;
	double UnDepth  = 0.0;
	double BdDepth  = 0.0;

	FindRec(InFile, "LOADMODE");
	fscanf(InFile, "%d",  &h_load->stage);
	FindRec(InFile, "packing_TIME");
	fscanf(InFile, "%lf", &h_load->tPacking);	// packing time


	//*-------- force option ------------*//
	char FlagCap[5];
	FindRec(InFile, "FoceCapillary");
	fgets(FlagCap, 5, InFile);
	if (strncmp(FlagCap, "Yes", 1)==0 || strncmp(FlagCap, "yes", 1)==0) 
	{ 
		*ForceCap = 1; 
	}


	FindRec(InFile,"REALSIZE");
	fscanf(InFile, "%lf", InitParSize);  // always the largest particle diameter in system
	double pSizeO = *InitParSize;

	

	FindRec(InFile,"CUTGAPFORNEIGHBORLIST");  // dimensionless 
	fscanf(InFile, "%lf", &h_dem->cutGap);
	


	FindRec(InFile, "EXPANDSIZE");
	fscanf(InFile, "%lf", &exComp);				// for computing as real size

	
	

	//*------- Boundary Condition -------*//
	/*------------------------------------*/
	char bounding[20];
	FindRec(InFile,"ContainerType");
	fgets(bounding, 20, InFile);
	

	if (strncmp(bounding, "Cylinder", 8)==0)
	{
		// allocation for host & device
		h_Cir =(CylinderBC *)malloc(sizeof(CylinderBC));
		cudaMalloc((void **)&d_Cir, sizeof(CylinderBC));
		// Container information 
		double Height = 0.0;
		FindRec(InFile,"CylinderBC");
		// center of bottom wall
		fscanf(InFile, "%lf", &h_Cir->cir.x);
		fscanf(InFile, "%lf", &h_Cir->cir.y);
		fscanf(InFile, "%lf", &h_Cir->Bw);
		fscanf(InFile, "%lf", &h_Cir->R);  // radius  
		fscanf(InFile, "%lf", &Height);    // height
		h_Cir->Tw = h_Cir->Bw + Height;
	}


	//*------- computation zone ---------*//
	/*------------------------------------*/
	// wrap all the zone
	double3 BotCorner = make_double3(h_Cir->cir.x - h_Cir->R, h_Cir->cir.y - h_Cir->R, h_Cir->Bw);
	double3 TopCorner = make_double3(h_Cir->cir.x + h_Cir->R, h_Cir->cir.y + h_Cir->R, h_Cir->Tw);

	
	//*------ FeedZone Zone ------*//
    FindRec(InFile,"FeedZone");
	fscanf(InFile, "%lf", &Feedcube->Lmin);
	fscanf(InFile, "%lf", &Feedcube->Lmax);

	
	//*------- Material Property --------*//
	/*------------------------------------*/
	FindRec(InFile,"MATERIAL");
	fscanf(InFile, "%lf", &h_Mat[0].density);
	fscanf(InFile, "%lf", &h_Mat[0].ymod);
	fscanf(InFile, "%lf", &h_Mat[0].pois);
	fscanf(InFile, "%lf", &h_Mat[0].sfrc);
	fscanf(InFile, "%lf", &h_Mat[0].rfrc);
	fscanf(InFile, "%lf", &h_Mat[0].dmpn);
	fscanf(InFile, "%lf", &h_Mat[0].yldp);

	fscanf(InFile, "%lf", &h_Mat[1].density);
	fscanf(InFile, "%lf", &h_Mat[1].ymod);
	fscanf(InFile, "%lf", &h_Mat[1].pois);
	fscanf(InFile, "%lf", &h_Mat[1].sfrc);
	fscanf(InFile, "%lf", &h_Mat[1].rfrc);
	fscanf(InFile, "%lf", &h_Mat[1].dmpn);
	fscanf(InFile, "%lf", &h_Mat[1].yldp);

	h_Mat[0].emod = h_Mat[0].ymod / (1-h_Mat[0].pois * h_Mat[0].pois);
	h_Mat[1].emod = h_Mat[1].ymod / (1-h_Mat[1].pois * h_Mat[1].pois);


	printf("h_Mat[0].emod: %lf.  \n", h_Mat[0].emod);




	//*--- Capillary force ---*//
	FindRec(InFile,"LIQUID");   
	fscanf(InFile, "%lf", &h_lqd->stension);
	fscanf(InFile, "%lf", &h_lqd->vol);
	fscanf(InFile, "%lf", &h_lqd->gapmn);
	

	//*--- Simulation parameters ---*//
	int nOut = 0;
	int nDump = 0;
	
	FindRec(InFile,"SIMULATION");  
	fscanf(InFile, "%lf", &h_dem->dtFactor);       // reduce time-step by user (0.5~1.0)
	fscanf(InFile, "%d",  &h_load->nOutPacking);


	


	h_dem->simTime  = 0.0;
	h_dem->dumptime = 0.0;
	h_dem->outtime  = 0.0;                          // when start to output
	h_dem->ctime    = 0.0;
	h_dem->iter     = 0;
	h_dem->fdtime   = 0.0;
	h_dem->Outs     = 0;




	//*--- Bond micro-properties ---*//
	/*-------------------------------*/
	double BcoRad    = 0.0;
	double ByMod     = 0.0;
	double BstfRatio = 0.0;
	double BtenStrn  = 0.0;
	double BshrStrn  = 0.0;


	/*--------------------------------------*/
	//*--------- Particle Diameter --------*//
	/*--------------------------------------*/
	char dfile[20];
	strcpy(dfile, genfile);
	strcat(dfile,"_dia.dat");  
	FILE *DiaFile;   
	char *mdia = "rt";
	DiaFile = fopen(dfile, mdia);
	if (DiaFile != NULL)
	{
		fclose(DiaFile);
		printf("reading diameter: \n");
		DiaInput(dfile, hRad, NP);
	}
	else     // set diameter to standard size
	{
		double Pdia = pSizeO;
		for(int i=0; i<NP; i++)  
		{
			hRad[i] = Pdia/2;
		}
	}



	


	//*--- determine Time Step ------------*//
	double pwemod = 2.0f * h_Mat[0].emod * h_Mat[0].emod / (h_Mat[0].emod + h_Mat[0].emod);
	double Rad_Min = hRad[0];     
	double Rad_Max = hRad[0]; 

	for(int i=1; i<NP; i++)
	{
		if(hRad[i] <= Rad_Min)
		{
			Rad_Min = hRad[i];    
		}

		if (hRad[i] >= Rad_Max)
		{
			Rad_Max = hRad[i];
		}
	}

	double sizeMn = 2.0 * Rad_Min;      
	double stiffk = 1.333 * pwemod * sqrt(sizeMn * 0.25);   
	double massMn = h_Mat[0].density * PI * pow(sizeMn, 3.0) / 6.0;  
	h_dem->dt = h_dem->dtFactor * PI * sqrt(0.5 * massMn / stiffk) * pow(10.0, -0.2); 


	
	// reading size level
	fprintf(LogFile, "\n\n=============================\n");
	fprintf(LogFile, "<*--- Particle Size Level ---*>\n");
	fprintf(LogFile, "=======================================================\n");
	fprintf(LogFile, "<Level_num>: %d \n", nLevels);
	char Flag[5];
	FindRec(InFile, "SIZELEVEL_INPUT");
	fgets(Flag, 5, InFile);
	if (strncmp(Flag, "Yes", 1)==0 || strncmp(Flag, "yes", 1)==0) 
	{
		for (int i=0; i<nLevels; i++)
		{
			/* input level is based on diameter while reduced to radius */
			fscanf(InFile, "%lf", &h_Level_size[i]);
			h_Level_size[i] = h_Level_size[i] / 2;
			fprintf(LogFile, "<level_%d>:  %lf\n",i, h_Level_size[i]*2);
		}
	}
	else
	{
		double lvl_gap = (Rad_Max - Rad_Min) / nLevels;
		for (int i=0; i<nLevels; i++)
		{
			h_Level_size[i] = lvl_gap * (i+1) + Rad_Min;
			fprintf(LogFile, "<level_%d>: %lf\n",i, h_Level_size[i]);
		}
	}

	
	


	for (int i=0; i<nLevels; i++)
	{
		double Sratio = h_Level_size[i] / Rad_Min;
		h_level_nNebors[i] = (int)ceil(1.2 * (pow((Sratio + 2 + h_dem->cutGap), 3) - pow(Sratio, 3)));

		h_level_nPars[i]   = 0;

		printf("h_level_nNebors[%d]:  %d.\n", i, h_level_nNebors[i]);
	}


	// determine which level particle belongs to
	for (int i=0; i<NP; i++)
	{
		if (hRad[i] <= h_Level_size[0])
		{
			h_pLevel[i]	   = 0;
			h_Nebor_Idx[i] = h_level_nPars[0];
			/* number of particles in the level */
			h_level_nPars[0] = h_level_nPars[0] + 1;

		}
		else
		{
			for (int iL = 1; iL<nLevels; iL++)
			{
				if (hRad[i] > h_Level_size[iL - 1] && hRad[i] <= h_Level_size[iL])
				{
					h_pLevel[i]    = iL;
					h_Nebor_Idx[i] = h_level_nPars[iL];
					/* number of particles in the level */
					h_level_nPars[iL] = h_level_nPars[iL] + 1;
				}
			}
		}
	}


	

	printf("h_Nebor_Idx[i]:    %d.\n", NP-1, h_Nebor_Idx[NP-1]);
	printf("h_level_nPars[0]:  %d.\n", h_level_nPars[0]);

	


	

	/*--------------------------------------*/
	//*---------- computing zone ----------*//
	/*--------------------------------------*/
	exComp = exComp * Rad_Max;
	h_Params->Origin.x = BotCorner.x - exComp;           //real size
	h_Params->Origin.y = BotCorner.y - exComp;
	h_Params->Origin.z = BotCorner.z - exComp;

	h_Params->EndPoint.x = TopCorner.x + exComp;
	h_Params->EndPoint.y = TopCorner.y + exComp;
	h_Params->EndPoint.z = TopCorner.z + exComp;




	// cutGap for neighbor reconstruction
	h_dem->cutGap = h_dem->cutGap * (Rad_Min / Rad_Max);
	printf("cutGap: %lf. \n", h_dem->cutGap);
	// after cell size convert to real size
	for (int i=0; i<nLevels; i++)
	{
		// size of cell is enlarged compared to level_size
		h_cell.Cellsize[i] = (1.0 + h_dem->cutGap*1.4) * 2.0 * h_Level_size[i];        // radius based          // real size
		printf("Level[%d]_cellsize:%lf. \n", i, h_cell.Cellsize[i]);
		h_cell.gridSize[i].x = (int)ceil( (h_Params->EndPoint.x - h_Params->Origin.x) / h_cell.Cellsize[i] );
		h_cell.gridSize[i].y = (int)ceil( (h_Params->EndPoint.y - h_Params->Origin.y) / h_cell.Cellsize[i] );
		h_cell.gridSize[i].z = (int)ceil( (h_Params->EndPoint.z - h_Params->Origin.z) / h_cell.Cellsize[i] );
		h_cell.numCells[i] = h_cell.gridSize[i].x * h_cell.gridSize[i].y * h_cell.gridSize[i].z;
	}
	
	

	
	



	fclose(InFile);
	////////////////////////////////////////////////////////////////////////////////////






	fprintf(LogFile, "<time step>: %lf e-6s \n", h_dem->dt * 1.0e6);
	fprintf(LogFile, "-------------------------------------------------------\n");
	if (*ForceCap == 1)
		fprintf(LogFile, "<Capillary Force>: Yes\n");
	else
		fprintf(LogFile, "<Capillary Force>: No\n");
	fprintf(LogFile, "=======================================================\n\n\n");


	fprintf(LogFile, "\n\n=============================\n");
	fprintf(LogFile, "<*--- Boundary Condition---*>\n");
	fprintf(LogFile, "=======================================================\n");
	fprintf(LogFile, "<BC type>: Cylinder\n");
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<circle>: (%lf, %lf)\t", h_Cir->cir.x, h_Cir->cir.y);
	fprintf(LogFile, "<radius>:  %lf\n", h_Cir->R);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<buttom>:  %lf\t\t", h_Cir->Bw);
	fprintf(LogFile, "<top>:  %lf\n", h_Cir->Tw);
	fprintf(LogFile, "=======================================================\n\n\n");

	

	fprintf(LogFile, "\n\n=====================\n");
	fprintf(LogFile, "<*--- Feed Zone ---*>\n");
	fprintf(LogFile, "=======================================================\n");
	fprintf(LogFile, "<lower >:  %lf\n", Feedcube->Lmin);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<upper >:  %lf\n", Feedcube->Lmax);
	fprintf(LogFile, "=======================================================\n\n\n");
	


	fprintf(LogFile, "\n\n=============================\n");
	fprintf(LogFile, "<*--- Material Property ---*>\n");
	fprintf(LogFile, "=======================================================\n");

	fprintf(LogFile, ">>PARTICLE<< \t\t\t");
	fprintf(LogFile, ">>WALL<<\n");
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<density>: %8.1f \t\t", h_Mat[0].density);
	fprintf(LogFile, "<density>: %8.1f \n", h_Mat[1].density);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<ymod>: %6.2e \t\t", h_Mat[0].ymod);
	fprintf(LogFile, "<ymod>: %6.2e \n", h_Mat[1].ymod);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<pois>: %6.2e \t\t", h_Mat[0].pois);
	fprintf(LogFile, "<pois>: %6.2e \n", h_Mat[1].pois);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<sfrc>: %6.2e \t\t", h_Mat[0].sfrc);
	fprintf(LogFile, "<sfrc>: %6.2e \n", h_Mat[1].sfrc);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<rfrc>: %6.2e \t\t", h_Mat[0].rfrc);
	fprintf(LogFile, "<rfrc>: %6.2e \n", h_Mat[1].rfrc);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<dmpn>: %6.2e \t\t", h_Mat[0].dmpn);
	fprintf(LogFile, "<dmpn>: %6.2e \n", h_Mat[1].dmpn);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<yldp>: %6.2e \t\t", h_Mat[0].yldp);
	fprintf(LogFile, "<yldp>: %6.2e \n", h_Mat[1].yldp);
	fprintf(LogFile, "-------------------------------------------------------\n");
	fprintf(LogFile, "<emod>: %6.2e \t\t", h_Mat[0].emod);
	fprintf(LogFile, "<emod>: %6.2e \n", h_Mat[1].emod);
	fprintf(LogFile, "=======================================================\n\n\n");

	fclose(LogFile);



	


}






