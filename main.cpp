#include <time.h>
#include <assert.h>
#include <math.h>
#include "heydem.h"
#include "double_math.h"
#include "thrust/scan.h"



int main(void)
{

	/*--------------------------------------------------------*/
	/*------------------- Global Variables ------------------ */
	/*--------------------------------------------------------*/
	int NP = 0;
	int nLevels = 1;                          

	int nBatch = 1;						// Feed parameters
	int TotalParticle = 0; 
	int nStlWallCnt = 0;				// Contact number

	double InitParSize=0.0;				// Particle Initial Info
	double SampleSize =0.0;				// Packing Sample Size 
	double CurHigh = 0.0;				// Current height
	// Force Control Flag
	int ForceCap  = 0;
	int RESTART   = 0;					// Flag for Restart
	int stage     = 1;					// computing stage
	int FeedFlag       = 0;
	int nReBldNebList  = 0;

	// Project Name
	char* genfile;
	genfile = (char *)malloc(20 * sizeof(char));  
	/*--------------------------------------------------------*/
	/*--------------------- Program Set --------------------- */
	/*--------------------------------------------------------*/
	FILE *TestFile; 
	char ProjName[20];
	char DumpName[20];


	// project name 
	strcpy(ProjName, "heyi");
	strcpy(DumpName, ProjName);
	strcpy(genfile,  ProjName);      
	strcat(ProjName, ".in");

	TestFile = fopen(ProjName, "rt");
	if (TestFile == NULL)
		exit(1);
	else
		fclose(TestFile);

	
	strcat(DumpName, "_dump.dat");
    TestFile = fopen(DumpName, "rt");

	if (TestFile == NULL)
	{
		// if dump file not exist then read input file
		FILE *SysFile = fopen(ProjName, "rt");

		FindRec(SysFile,"NUMPARTICLE");
		fscanf(SysFile, "%d", &NP);
		FindRec(SysFile,"SIZELEVEL");
		fscanf(SysFile, "%d", &nLevels);
		printf("nLevels: %d. \n", nLevels);
		fclose(SysFile);
	}
	else
	{
		fclose(TestFile);
		RESTART = 1;
		// Load parameters for memory allocation
	    LoadPara(DumpName, &NP, &nBatch, &TotalParticle);

		FILE *SysFile;   
		SysFile = fopen(ProjName, "rt");
		FindRec(SysFile,"SIZELEVEL");
		fscanf(SysFile, "%d", &nLevels);
		printf("nLevels: %d. \n", nLevels);
		fclose(SysFile);
	}




	/*--------------------------------------------------------*/
	/*--------------------- Program Log --------------------- */
	/*--------------------------------------------------------*/
	FILE *LogFile; 
	char LogName[20];
	strcpy(LogName, genfile);
	strcat(LogName ,"_Log.dat"); 
	LogFile = fopen(LogName, "rt");
	if (RESTART == 0)
	{

		LogFile = fopen(LogName, "wt");
		fprintf(LogFile, "==================|*--- PROJECT---*|===================\n");
		fprintf(LogFile, "<File>: new\n");
		// system time
		time_t timep;
		time(&timep);
		fprintf(LogFile, "<Time>: %s",   ctime(&timep));
		fprintf(LogFile, "<Name>: %s\n", genfile);
		fprintf(LogFile, "=======================================================\n\n\n");

		
		fprintf(LogFile, "\n\n=================================\n");
		fprintf(LogFile,   "<*--- Simulation Parameters ---*>\n");
		fprintf(LogFile, "=======================================================\n");
		fprintf(LogFile, "<Particle>: %d \n", NP);
		fprintf(LogFile, "-------------------------------------------------------\n");
	}
	else
	{

		LogFile = fopen(LogName, "wt");
		fprintf(LogFile, "\n\n==============|*--- PROJECT ---*|===============\n");
		fprintf(LogFile, "<File>: restarted\n");	

		time_t timep;    // system time
		time(&timep);
		fprintf(LogFile, "<Time>: %s",        ctime(&timep));
		fprintf(LogFile, "<Name>: %s\n",      genfile);
		fprintf(LogFile, "=======================================================\n\n\n");

		fprintf(LogFile, "\n\n=================================\n");
		fprintf(LogFile,   "<*--- Simulation Parameters ---*>\n");
		fprintf(LogFile, "=======================================================\n");
		fprintf(LogFile, "<Particle>: %d \t", NP);
		fprintf(LogFile, "-------------------------------------------------------\n");
		fprintf(LogFile, "<Feed batch>: %d \t",       nBatch);
		fprintf(LogFile, "<Current particle>: %d \n", TotalParticle);
		fprintf(LogFile, "-------------------------------------------------------\n");
	}
	fclose(LogFile);




	/*--------------------------------------------------------*/
	/* ----------------- Generic Definition ----------------- */
	/*--------------------------------------------------------*/
	LOADMODE *h_load = (LOADMODE *)malloc(sizeof(LOADMODE));

	

	/*-------------------------------------------------*/
    /*------ Allocation: particle Info ----------------*/
	/*-------------------------------------------------*/
	ParticleInfoAlloc(NP);


	

	/*-------------------------------------------------*/
	/*----------- Allocation: feed zone ---------------*/
	/*-------------------------------------------------*/
	FeedZone *Feedcube;
	FeedZoneAlloc(Feedcube);     

	


	


	/*-------------------------------------------------*/
	/*------ Particle size levels ---------------------*/
	/*-------------------------------------------------*/ 
	double *h_Level_size = (double *)malloc(nLevels * sizeof(double));		// maximum particle size of each level
	int *h_level_nNebors = (int *)malloc(nLevels * sizeof(int));			/* maximum number of neighbors per level */
	int *h_level_nPars	 = (int *)malloc(nLevels * sizeof(int));			/* number of particles in each level */
	int *h_Nebor_Idx     = (int *)malloc(NP * sizeof(int));
	int *h_pLevel        = (int *)malloc(NP * sizeof(int));					/* which level the particle belongs to */

	memset(h_Level_size,	0,	nLevels * sizeof(double)); 
	memset(h_level_nNebors, 0,	nLevels * sizeof(int)); 
	memset(h_level_nPars,	0,	nLevels * sizeof(int));
	memset(h_Nebor_Idx,		0,	NP * sizeof(int));		
	memset(h_pLevel,		-1, NP * sizeof(int));

	__device__ double *d_Level_size;
	__device__ int	  *d_level_nNebors;
	__device__ int    *d_level_nPars;
	__device__ int    *d_pLevel;
	__device__ int    *d_Nebor_Idx;

	cutilSafeCall(cudaMalloc((void **)&d_Level_size,    nLevels * sizeof(double)));
	cutilSafeCall(cudaMalloc((void **)&d_level_nNebors, nLevels * sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_level_nPars,   nLevels * sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_pLevel,        NP * sizeof(int)));			
	cutilSafeCall(cudaMalloc((void **)&d_Nebor_Idx,     NP * sizeof(int)));



	//*------------ Allocation: Parameters -------------*//
	SIMULATION	*h_dem;
	UNIT		*Rdu;       
	int			*h_lqdCnt;
	MatType		*h_Mat;									// material properties
	LIQUID		*h_lqd;									// capillary force parameters
	PARAMETER	*h_Params;								// Other computational parameters
	SEARCHCELL	h_cell;
	__device__ int	*d_lqdCnt;
	__device__ int	*d_lqdCnt_Old;
	__device__ SEARCHCELL d_cell;	

	ParamAlloc(h_dem, Rdu,   h_lqdCnt, d_lqdCnt, d_lqdCnt_Old,
		       h_Mat, h_lqd, h_Params, h_cell, d_cell, nLevels, NP);




	
	/*-------------------------------------------------*/
    /* ---------- Initialization / Read Data --------- */
	/*-------------------------------------------------*/
	CylinderBC *h_Cir;									// BC type choose & allocation 
	__device__ CylinderBC *d_Cir;


	ReadData(h_Level_size, h_pLevel, h_level_nNebors, h_level_nPars, h_Nebor_Idx,
		     h_Cir,     d_Cir,  Feedcube, h_Mat, h_lqd, 
		     h_Params,  &InitParSize, 
			 &ForceCap, h_dem, h_load, hRad, h_cell,
			 NP, nLevels, genfile, RESTART);
	


	



	/*-------------------------------------------------*/
	/*------------ Neighbor List  USE -----------------*/
	/*-------------------------------------------------*/
	/*--- particle ---*/
	int *h_Num_Nebor;
	int *h_1DNebor;   
	__device__ int	*d_Num_Nebor;
	__device__ int	*d_Num_Nebor_Old; 
	__device__ int	*d_Nebor1;
	__device__ int	*d_OldNebor1;        
	__device__ double	*d_pMoveforReBld;					// rebuild flag variable


	/*--- stationary wall ---*/
	int *h_Num_stlWall, *h_StlWList_1D;  
	__device__ int *d_Num_StlWall,  *d_Num_StlWall_Old;
	__device__ int *d_StlW_List,    *d_StlW_List_Old;     
	size_t pitch_Sw1, pitch_Sw2;		




	/*-------------------------------------------------*/
	/*------ Allocation: contact history --------------*/
	/*-------------------------------------------------*/ 
	int StlMaxWallPerBall = 3;
	

	PPCONTACT_HISTORY h_cntHstry_pp;
	PWCONTACT_HISTORY h_cntHstry_Stlpw;
	__device__ PPCONTACT_HISTORY d_cntHstry_pp;
	__device__ PPCONTACT_HISTORY d_cntHstry_ppOld;
	__device__ PWCONTACT_HISTORY d_cntHstry_Stlpw;
	__device__ PWCONTACT_HISTORY d_cntHstry_Stlpw_Old;

	// total number of particle neighbors
	int nPPCnt = 0;
	for (int i=0; i<nLevels; i++)
	{
		nPPCnt = nPPCnt + h_level_nPars[i] * h_level_nNebors[i];
	}
	

	/* total number of wall neighbors */
	int nWallCnt = NP * StlMaxWallPerBall;




  





	/*-------------------------------------------------*/
	/*--- Allocate particle contact history -----------*/
	/*-------------------------------------------------*/
	//*--- allocation and initialization ---*//
	NeborListAlloc(h_Num_Nebor,   h_1DNebor, d_pMoveforReBld, 
		           d_Num_Nebor,   d_Num_Nebor_Old, d_Nebor1,      d_OldNebor1, 
		           h_Num_stlWall, h_StlWList_1D,   d_Num_StlWall, d_Num_StlWall_Old,
		           d_StlW_List,   d_StlW_List_Old, pitch_Sw1,	  pitch_Sw2,
		           NP, nPPCnt, StlMaxWallPerBall);

	PPCntHstry_Alloc(h_cntHstry_pp, d_cntHstry_pp, d_cntHstry_ppOld, nPPCnt);
	WallCntHstry_Alloc(h_cntHstry_Stlpw, d_cntHstry_Stlpw, d_cntHstry_Stlpw_Old, nWallCnt);



	/* ----------------------------------------------- */
	/* --------------- Dimensionless ----------------- */
	/* ----------------------------------------------- */
	UnitReduce(Rdu, h_Cir, Feedcube, h_load, hRad, h_Mat, 
		       h_lqd, InitParSize, h_dem, h_Params, h_cell, 
			   h_Level_size, nLevels, NP, RESTART, genfile);



	// neighbor index for each particle
	cutilSafeCall(cudaMemcpy(d_Level_size,    h_Level_size,	   nLevels * sizeof(double), cudaMemcpyHostToDevice));  
	cutilSafeCall(cudaMemcpy(d_level_nNebors, h_level_nNebors, nLevels * sizeof(int),	 cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_level_nPars,   h_level_nPars,   nLevels * sizeof(int),	 cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_pLevel,		  h_pLevel,		   NP * sizeof(int),		 cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_Nebor_Idx,     h_Nebor_Idx,	   NP * sizeof(int),		 cudaMemcpyHostToDevice));  

	/*-------------------------------------------------*/
	/* --------- Parameter for contact force --------- */
	/*-------------------------------------------------*/
	calParaPP *h_cal = (calParaPP*)malloc(sizeof(calParaPP));
	memset(h_cal, 0, sizeof(calParaPP));
	h_cal->emod     = (h_Mat[0].emod * h_Mat[0].emod)/(h_Mat[0].emod + h_Mat[0].emod); 
	h_cal->pois     = 2.0 * (h_Mat[0].pois * h_Mat[0].pois)/(h_Mat[0].pois + h_Mat[0].pois);
	h_cal->sfrc     = h_Mat[0].sfrc;
	h_cal->ystress  = h_Mat[0].yldp * h_Mat[0].yldp / (h_Mat[0].yldp + h_Mat[0].yldp);  //h_cal->ystress  = 2.0 * h_Mat[0].yldp * h_Mat[0].yldp / (h_Mat[0].yldp + h_Mat[0].yldp);  
	h_cal->ydsp_t   = pow((PI * h_cal->ystress / (2.0 * h_cal->emod)), 2.0);
	h_cal->yforce_t = 4.0 / 3.0 * h_cal->emod * pow(h_cal->ydsp_t, 1.5);
	h_cal->ystiff_t = PI * h_cal->ystress;



	calParaPW *h_pWcal = (calParaPW*)malloc(sizeof(calParaPW));
	memset(h_pWcal, 0, sizeof(calParaPW));
	h_pWcal->emod     = h_Mat[0].emod;                                              //h_pWcal->emod = (h_Mat[0].emod * h_Mat[1].emod)/(h_Mat[0].emod + h_Mat[1].emod); 
	h_pWcal->pois     = 2.0 * (h_Mat[0].pois * h_Mat[1].pois)/(h_Mat[0].pois + h_Mat[1].pois);
	h_pWcal->sfrc     = pow(h_Mat[1].sfrc * h_Mat[1].sfrc, 0.5);                    // change for Effect of inter-particle friction coefficient
	h_pWcal->ystress  = h_Mat[0].yldp;                                              //h_pWcal->ystress  = 2 * h_Mat[0].yldp * h_Mat[1].yldp / (h_Mat[0].yldp + h_Mat[1].yldp);  
	h_pWcal->ydsp_t   = pow((PI * h_pWcal->ystress / (2.0 * h_pWcal->emod)), 2.0);
	h_pWcal->yforce_t = 4.0 / 3.0 * h_pWcal->emod * pow(h_pWcal->ydsp_t, 1.5);
	h_pWcal->ystiff_t = PI * h_pWcal->ystress;
	


	/*-------------------------------------------------*/
	/* ----- Parameters Transfer: Host To Device ----- */
	/*-------------------------------------------------*/
	//*--- computing parameters constant---*//
	setMaterial(h_Mat);                     // material properties--constant memory
	setLiquid(h_lqd);                       // liquid properties
	setParameters(h_Params); 
	setCalpara(h_cal, h_pWcal);             // contact force parameters


	// Parameters
	cudaMemcpy(dRad, hRad, NP * sizeof(double),	cudaMemcpyHostToDevice);   
	cudaMemcpy(d_cell.Cellsize, h_cell.Cellsize, nLevels * sizeof(double),	cudaMemcpyHostToDevice); 
	cudaMemcpy(d_cell.numCells, h_cell.numCells, nLevels * sizeof(int),		cudaMemcpyHostToDevice); 
	cudaMemcpy(d_cell.gridSize, h_cell.gridSize, nLevels * sizeof(int3),	cudaMemcpyHostToDevice); 



	//*-------------------------------------------------*/
	//*------ Allocation: cell search use --------------*/
	//*-------------------------------------------------*/
	int *h_Ncell_Idx = (int *)malloc(nLevels * sizeof(int));
	memset(h_Ncell_Idx, 0, nLevels * sizeof(int));

	__device__ int *d_Ncell_Idx;

	cutilSafeCall(cudaMalloc((void **)&d_Ncell_Idx,  nLevels * sizeof(int)));
	IndexOffset(d_cell.numCells, d_Ncell_Idx, nLevels);
	cutilSafeCall(cudaMemcpy(h_Ncell_Idx,  d_Ncell_Idx, nLevels*sizeof(int), cudaMemcpyDeviceToHost)); 


	// allocate
	int Sys_nCells = h_Ncell_Idx[nLevels-1];                   // CPU data--total cell number
	SearchCellAlloc(NP, Sys_nCells);


	/*-------------------------------------------------*/
	/*------------- creating time events --------------*/
	/*-------------------------------------------------*/
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	
	/*-------------------------------------*/
	//*--- Boundary Condition transfer ---*//
	/*-------------------------------------*/
	h_dem->simTime     = h_load->tPacking;	
	h_dem->outinterval = h_dem->simTime / h_load->nOutPacking;
	h_dem->fdinterval  = 1.0 * h_dem->outinterval; 
	cutilSafeCall(cudaMemcpy(d_Cir, h_Cir, sizeof(CylinderBC), cudaMemcpyHostToDevice));
	

	/*-------------------------------------------------*/
    /* ------ Main Loop & Calculation on device ------ */
	/*-------------------------------------------------*/
	int *h_ReNebld = (int *)malloc(sizeof(int));
	memset(h_ReNebld, 0, sizeof(int));
	__device__ int* d_ReNebld;
	cutilSafeCall(cudaMalloc((void**)&d_ReNebld, sizeof(int)));
	cutilSafeCall(cudaMemset(d_ReNebld, 0, sizeof(int)));


	LogFile = fopen(LogName, "a");
	fprintf(LogFile,"=======================================================\n");
	fprintf(LogFile,"||||||||||||||||||*--- Main Loop ---*||||||||||||||||||\n");
	fprintf(LogFile,"=======================================================\n\n");
	fflush(LogFile);
	cutilSafeCall(cudaGetLastError());


	do{

		if (h_dem->iter == 0 || RESTART == 1)
			h_dem->outtime = h_dem->dt;           // output first status for each stage
		else
			h_dem->outtime = h_dem->ctime + h_dem->outinterval;


        do{
			
			cutilSafeCall(cudaGetLastError());
			/*-------------------------------------------------------*/
			/*--------------- Particle Generation -------------------*/
			/*-------------------------------------------------------*/
			if (TotalParticle < NP && h_load->stage == 1 && h_dem->ctime >= h_dem->fdtime)          
			{

				FeedFlag = 1;
				cutilSafeCall(cudaMemcpy(hPos, dPos, NP * sizeof(double3), cudaMemcpyDeviceToHost));
				TotalParticle = ParticleFeed(Feedcube, h_Cir, InitParSize, hPos, hRad,TotalParticle, NP); 

				nBatch = nBatch + 1;
				h_dem->fdtime = h_dem->fdtime + 1 * h_dem->fdinterval;
				cutilSafeCall(cudaMemcpy(dPos, hPos, NP * sizeof(double3), cudaMemcpyHostToDevice));   			
				
				if (TotalParticle == NP)
				{
					/*--- Information Log ---*/
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&elapsedTime, start, stop);
					
					fprintf(LogFile,"\n<<< Feed Over >>> |*---time:%10.4lf---*|\n\n", elapsedTime/1000);
					fflush(LogFile);
				}
				cutilSafeCall(cudaGetLastError());
			}
			
			
			
			cutilSafeCall(cudaGetLastError());
			/*-------------------------------------------------------*/
			/*------- Neighbor list Reconstruction ------------------*/
			/*-------------------------------------------------------*/
            //*--- Criterion for Rebuild ---*//
			cutilSafeCall(cudaMemcpy(h_ReNebld, d_ReNebld, sizeof(int), cudaMemcpyDeviceToHost));
			
			//*--- Reconstruction ---*//
		    if ( *h_ReNebld > 0 || FeedFlag == 1 || RESTART == 1)   
			{
				// last step interaction information copy   
				if (h_dem->iter != 0 || RESTART == 1) 
				{
					/*--- particle neighbor ---*/
					cutilSafeCall(cudaMemcpy(d_Num_Nebor_Old, d_Num_Nebor, NP * sizeof(int),     cudaMemcpyDeviceToDevice));
					cutilSafeCall(cudaMemcpy(d_OldNebor1,     d_Nebor1,    nPPCnt * sizeof(int), cudaMemcpyDeviceToDevice));
					PPCntHstryCopy(d_cntHstry_ppOld, d_cntHstry_pp, nPPCnt);

					/*---stationary wall neighbor ---*/
					cutilSafeCall(cudaMemcpy(d_Num_StlWall_Old, d_Num_StlWall, NP * sizeof(int), cudaMemcpyDeviceToDevice));
					cutilSafeCall(cudaMemcpy2D(d_StlW_List_Old, pitch_Sw2,	 d_StlW_List, pitch_Sw1, StlMaxWallPerBall * sizeof(int), NP,cudaMemcpyDeviceToDevice));
					WallCntHstryCopy(d_cntHstry_Stlpw_Old, d_cntHstry_Stlpw, nWallCnt);
				}
				
				

				// build neighbor list for particle
				cutilSafeCall(cudaMemset(d_pMoveforReBld, 0, NP * sizeof(double)));


				BldNeborList(d_GridParticleHash, d_GridParticleIndex, d_CellStart, d_CellEnd,
					         d_pLevel, d_cell, d_Ncell_Idx,  d_Level_size, 
							 d_level_nPars, d_level_nNebors, nLevels,
					         dPos, Sys_nCells, d_Nebor1, d_Nebor_Idx, d_Num_Nebor, 
						     TotalParticle, dRad); 

				
				// build neighbor wall list for particle
				BldCirWall(dPos, dRad, d_Cir, d_StlW_List, pitch_Sw1, d_Num_StlWall, TotalParticle);
				
				
				//*--- History Copy ---*//		
				if (h_dem->iter != 0 || RESTART == 1)
				{

					HstryCopy(d_Nebor1, d_OldNebor1,
							  d_pLevel, d_level_nNebors, d_level_nPars, d_Nebor_Idx,
						      d_cntHstry_ppOld, d_cntHstry_pp, d_Num_Nebor, d_Num_Nebor_Old,
						      TotalParticle, NP);
					
					
					HstryCopyPW(d_StlW_List, d_StlW_List_Old, pitch_Sw1, pitch_Sw2,
						        d_cntHstry_Stlpw_Old,  d_cntHstry_Stlpw,
						        d_Num_StlWall, d_Num_StlWall_Old, TotalParticle, NP);
				}

				
				// reset flags for particle feed and restart
				FeedFlag = 0;
				RESTART  = 0;
				nReBldNebList = nReBldNebList + 1;
				*h_ReNebld = 0;
			}
			
			
			

			/*-------------------------------------------------------*/
			/*||||||||||||||||| Force Calculation |||||||||||||||||||*/
			/*-------------------------------------------------------*/
			// process variable initialization
			VarInit(dForce, dMom, dRMom, d_lqdCnt, d_lqdCnt_Old, NP);
			
			
			
			// contact & gravity & capillary force 
			PpInterCalc(dPos, dVel, dAngVel, dRad, d_Nebor1, 
						d_pLevel, d_level_nNebors, d_level_nPars, d_Nebor_Idx, 
				        d_cntHstry_pp, d_Num_Nebor, d_lqdCnt, d_lqdCnt_Old, dForce, dMom, dRMom,
				        TotalParticle, NP, ForceCap, h_load->stage);
			

			
			//*--- boundary condition ---*//
			CirPWCalc(dPos, dVel, dAngVel, dForce, dMom, dRMom, dRad, d_Cir, 
					  d_cntHstry_Stlpw, d_Num_StlWall, d_StlW_List, pitch_Sw1, 
					  TotalParticle, NP, h_load->stage);
			
			

			//*--- update & integration ---*/
			cutilSafeCall(cudaMemset(d_ReNebld, 0, sizeof(int)));
			UpdateSystem(dRad, dPos, dVel, dAngVel, dForce, dMom, dRMom, 
				         d_ReNebld, d_pMoveforReBld, TotalParticle);

			
			
			// step Forward
			h_dem->ctime = h_dem->ctime + h_dem->dt;
			h_dem->iter  = h_dem->iter + 1;


		// output file when it's first time 
	    }while (h_dem->ctime < h_dem->outtime);      
		
	
		
		// retrieve particle information back to CPU
		cutilSafeCall(cudaMemcpy(hPos,    dPos,    NP * sizeof(double3), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hVel,    dVel,    NP * sizeof(double3), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hAngVel, dAngVel, NP * sizeof(double3), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hForce,  dForce,  NP * sizeof(double3), cudaMemcpyDeviceToHost));			
		
		
		// Copy Neighbor list and Contact history to CPU
		/*NeborToCPU(h_Num_Nebor,   d_Num_Nebor,   h_1DNebor,     d_Nebor1, 
			       h_Num_stlWall, d_Num_StlWall, h_StlWList_1D, d_StlW_List,
			       h_cntHstry_pp, d_cntHstry_pp, h_cntHstry_Stlpw, d_cntHstry_Stlpw, 
			       NP, nPPCnt, pitch_Sw1, StlMaxWallPerBall);*/
		
		
		// Particle Information for display
		WrtTec(hPos, hVel, hForce, hAngVel, h_dem, hRad, Rdu, TotalParticle, NP);
		
		
		
		//*--- Information Log ---*//	
		fprintf(LogFile, "%-5d\t",            h_dem->Outs);
		fprintf(LogFile, "runtime:%-7.4lf\t", h_dem->ctime*Rdu->rtunit);
	    fflush(LogFile);
		// time recording
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cutilSafeCall(cudaGetLastError());
	
		if (TotalParticle < NP)
		{
			fprintf(LogFile,"elapsed:%-20.5fms\t",  elapsedTime);
			fprintf(LogFile,"rebuild:%-d\t",      nReBldNebList);
			fprintf(LogFile,"particle:%-d\t",     TotalParticle);
			fprintf(LogFile,"Iterate:%-d\n",      h_dem->iter);
			nReBldNebList = 0;
		}
		else
		{
			fprintf(LogFile,"elapsed:%-20.5fms\t", elapsedTime);
			fprintf(LogFile,"rebuild:%-d\t",      nReBldNebList);
			fprintf(LogFile,"Iterate:%-d\n",      h_dem->iter);
			nReBldNebList = 0;
		}

		fprintf(LogFile,"-------------------------------------------------------\n");
		fflush(LogFile);
		
		// output times
		h_dem->Outs = h_dem->Outs + 1; 

	}while(h_dem->ctime < h_dem->simTime);




	/*-------------------------------------------------*/
	/* ------------ Recording Time Usage ------------- */
	/*-------------------------------------------------*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(LogFile,"\nGPU DEM Time used: %-12.2lf s \n\n", elapsedTime/1000);
	fclose(LogFile);


	/*-------------------------------------------------*/
	/*-------------------- Cleanup --------------------*/
	/*-------------------------------------------------*/
	cudaFree(d_GridParticleHash);
	cudaFree(d_GridParticleIndex);
	cudaFree(d_CellStart);
	cudaFree(d_CellEnd);
	cudaFree(dPos);
	cudaFree(dVel);	
	cudaFree(dAngVel);
	cudaFree(dForce);
	cudaFree(dMom);
	cudaFree(dRMom);
	cudaFree(d_Nebor1);
	cudaFree(d_OldNebor1);
	cudaFree(d_Num_Nebor);
	cudaFree(d_lqdCnt);
	cudaFree(d_lqdCnt_Old);
	cudaFree(dRad);
	cudaFree(d_ReNebld);
	cudaFree(d_pMoveforReBld);
	cudaFree(d_Num_StlWall);
	cudaFree(d_Cir);

	free(genfile);
	free(hPos);
	free(hVel);
	free(hAngVel);
	free(hRad);
	free(hForce);
	free(h_Num_Nebor);
	free(h_Mat);
	free(h_lqd);
	free(h_dem);
	free(Rdu);
	free(h_1DNebor);
	free(h_Cir);
}
