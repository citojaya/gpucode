#include "heydem.h"


//*--- particle information ---*//
double3 *hPos, *hVel, *hAngVel, *hForce;
double *hRad;

__device__ double3 *dPos, *dVel, *dAngVel,  *dForce, *dMom; 
__device__ double  *dRMom, *dRad;




//*--- search cell ---*// 
__device__ int*  d_GridParticleHash;                // grid hash value for each particle
__device__ int*  d_GridParticleIndex;               // particle index for each particle
__device__ int*  d_CellStart;                       // index of start of each cell in sorted list
__device__ int*  d_CellEnd;                         // index of end of cell



void ParticleInfoAlloc(int NP)
{
	//*--- on host ---*//
	hPos    = (double3 *)malloc(NP * sizeof(double3));
	hVel    = (double3 *)malloc(NP * sizeof(double3));
	hForce  = (double3 *)malloc(NP * sizeof(double3));
	hAngVel = (double3 *)malloc(NP * sizeof(double3));
	hRad    = (double  *)malloc(NP * sizeof(double));
	memset(hPos,    0, NP * sizeof(double3));
	memset(hVel,    0, NP * sizeof(double3));
	memset(hForce,  0, NP * sizeof(double3));
	memset(hAngVel, 0, NP * sizeof(double3));
	memset(hRad,	0, NP * sizeof(double));

	//*--- on device ---*//
	cutilSafeCall(cudaMalloc((void**)&dPos,    NP * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&dVel,    NP * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&dAngVel, NP * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&dForce,  NP * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&dMom,    NP * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&dRMom,   NP * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&dRad,    NP * sizeof(double)));

	cutilSafeCall(cudaMemset(dPos,    0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dVel,    0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dAngVel, 0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dForce,  0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dMom,    0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dRMom,   0, NP * sizeof(double)));
	cutilSafeCall(cudaMemset(dRad,    0, NP * sizeof(double)));  

}



void WallCntHstry_Alloc(PWCONTACT_HISTORY& h_cntHstry_Stlpw, PWCONTACT_HISTORY& d_cntHstry_Stlpw, 
	                    PWCONTACT_HISTORY& d_cntHstry_Stlpw_Old, int nWallCnt)
{
	// host memory
	h_cntHstry_Stlpw.dispt   = (double3 *)malloc(nWallCnt * sizeof(double3));
	h_cntHstry_Stlpw.nDsp    = (double *)malloc(nWallCnt * sizeof(double));
	h_cntHstry_Stlpw.nDsp_mx = (double *)malloc(nWallCnt * sizeof(double));
	h_cntHstry_Stlpw.plstRad = (double *)malloc(nWallCnt * sizeof(double));
	h_cntHstry_Stlpw.plstDfm = (double *)malloc(nWallCnt * sizeof(double));
	h_cntHstry_Stlpw.stage   = (int *)malloc(nWallCnt * sizeof(int));
	h_cntHstry_Stlpw.lqdBrg  = (int *)malloc(nWallCnt * sizeof(int));

	memset(h_cntHstry_Stlpw.dispt,   0, nWallCnt * sizeof(double3));
	memset(h_cntHstry_Stlpw.nDsp,    0, nWallCnt * sizeof(double));
	memset(h_cntHstry_Stlpw.nDsp_mx, 0, nWallCnt * sizeof(double));
	memset(h_cntHstry_Stlpw.plstRad, 0, nWallCnt * sizeof(double));
	memset(h_cntHstry_Stlpw.plstDfm, 0, nWallCnt * sizeof(double));
	memset(h_cntHstry_Stlpw.stage,   0, nWallCnt * sizeof(int));
	memset(h_cntHstry_Stlpw.lqdBrg,  0, nWallCnt * sizeof(int));

	// device 
	cudaMalloc((void**)&d_cntHstry_Stlpw.dispt,   nWallCnt * sizeof(double3));
	cudaMalloc((void**)&d_cntHstry_Stlpw.nDsp,    nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw.nDsp_mx, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw.plstRad, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw.plstDfm, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw.stage,   nWallCnt * sizeof(int));
	cudaMalloc((void**)&d_cntHstry_Stlpw.lqdBrg,  nWallCnt * sizeof(int));

	cudaMemset(d_cntHstry_Stlpw.dispt,   0, nWallCnt * sizeof(double3));
	cudaMemset(d_cntHstry_Stlpw.nDsp,    0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw.nDsp_mx, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw.plstRad, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw.plstDfm, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw.stage,   0, nWallCnt * sizeof(int));
	cudaMemset(d_cntHstry_Stlpw.lqdBrg,  0, nWallCnt * sizeof(int));

	// device copy
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.dispt,   nWallCnt * sizeof(double3));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.nDsp,    nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.nDsp_mx, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.plstRad, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.plstDfm, nWallCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.stage,   nWallCnt * sizeof(int));
	cudaMalloc((void**)&d_cntHstry_Stlpw_Old.lqdBrg,  nWallCnt * sizeof(int));

	cudaMemset(d_cntHstry_Stlpw_Old.dispt,   0, nWallCnt * sizeof(double3));
	cudaMemset(d_cntHstry_Stlpw_Old.nDsp,    0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw_Old.nDsp_mx, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw_Old.plstRad, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw_Old.plstDfm, 0, nWallCnt * sizeof(double));
	cudaMemset(d_cntHstry_Stlpw_Old.stage,   0, nWallCnt * sizeof(int));
	cudaMemset(d_cntHstry_Stlpw_Old.lqdBrg,  0, nWallCnt * sizeof(int));
}




void PPCntHstry_Alloc(PPCONTACT_HISTORY& h_cntHstry_pp, PPCONTACT_HISTORY& d_cntHstry_pp, 
	                  PPCONTACT_HISTORY& d_cntHstry_ppOld, int nPPCnt)
{
	// host memory
	h_cntHstry_pp.dispt   = (double3 *)malloc(nPPCnt * sizeof(double3));
	h_cntHstry_pp.nDsp     = (double *)malloc(nPPCnt * sizeof(double));
	h_cntHstry_pp.nDsp_mx  = (double *)malloc(nPPCnt * sizeof(double));
	h_cntHstry_pp.plstRad  = (double *)malloc(nPPCnt * sizeof(double));
	h_cntHstry_pp.plstDfm  = (double *)malloc(nPPCnt * sizeof(double));
	h_cntHstry_pp.stage   = (int *)malloc(nPPCnt * sizeof(int));
	h_cntHstry_pp.lqdBrg  = (int *)malloc(nPPCnt * sizeof(int));


	memset(h_cntHstry_pp.dispt,   0, nPPCnt * sizeof(double3));
	memset(h_cntHstry_pp.nDsp,    0, nPPCnt * sizeof(double));
	memset(h_cntHstry_pp.nDsp_mx, 0, nPPCnt * sizeof(double));
	memset(h_cntHstry_pp.plstRad, 0, nPPCnt * sizeof(double));
	memset(h_cntHstry_pp.plstDfm, 0, nPPCnt * sizeof(double));
	memset(h_cntHstry_pp.stage,   0, nPPCnt * sizeof(int));
	memset(h_cntHstry_pp.lqdBrg,  0, nPPCnt * sizeof(int));


	// device 
	printf(" - nPPCnt: %d. \n", nPPCnt);
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.dispt,   nPPCnt * sizeof(double3)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.nDsp,     nPPCnt * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.nDsp_mx,  nPPCnt * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.plstRad,  nPPCnt * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.plstDfm,  nPPCnt * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.stage,   nPPCnt * sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_pp.lqdBrg,  nPPCnt * sizeof(int)));
	

	cutilSafeCall(cudaMemset(d_cntHstry_pp.dispt,   0, nPPCnt * sizeof(double3)));
	cutilSafeCall(cudaMemset(d_cntHstry_pp.nDsp,    0, nPPCnt * sizeof(double)));
	cudaMemset(d_cntHstry_pp.nDsp_mx, 0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_pp.plstRad, 0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_pp.plstDfm, 0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_pp.stage,   0, nPPCnt * sizeof(int));
	cudaMemset(d_cntHstry_pp.lqdBrg,  0, nPPCnt * sizeof(int));
	
	// device copy
	cutilSafeCall(cudaMalloc((void**)&d_cntHstry_ppOld.dispt,   nPPCnt * sizeof(double3)));
	cudaMalloc((void**)&d_cntHstry_ppOld.nDsp,     nPPCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_ppOld.nDsp_mx,  nPPCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_ppOld.plstRad,  nPPCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_ppOld.plstDfm,  nPPCnt * sizeof(double));
	cudaMalloc((void**)&d_cntHstry_ppOld.stage,   nPPCnt * sizeof(int));
	cudaMalloc((void**)&d_cntHstry_ppOld.lqdBrg,  nPPCnt * sizeof(int));


	cudaMemset(d_cntHstry_ppOld.dispt,   0, nPPCnt * sizeof(double3));
	cudaMemset(d_cntHstry_ppOld.nDsp,    0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_ppOld.nDsp_mx, 0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_ppOld.plstRad, 0, nPPCnt * sizeof(double));
	cudaMemset(d_cntHstry_ppOld.plstDfm, 0, nPPCnt * sizeof(double));
	cutilSafeCall(cudaMemset(d_cntHstry_ppOld.stage,   0, nPPCnt * sizeof(int)));
	cutilSafeCall(cudaMemset(d_cntHstry_ppOld.lqdBrg,  0, nPPCnt * sizeof(int)));

}






void NeborListAlloc(int*& h_Num_Nebor,   int*& h_1DNebor, double*& d_pMoveforReBld, 
	                int*& d_Num_Nebor,   int*& d_Num_Nebor_Old,
                    int*& d_Nebor1,      int*& d_Nebor1_Old, 
	                int*& h_Num_stlWall, int*& h_StlWList_1D, 
                    int*& d_Num_StlWall, int*& d_Num_StlWall_Old,
                    int*& d_StlW_List,   int*& d_StlW_List_Old,    
                    size_t& pitch_Sw1,   size_t& pitch_Sw2,
	                int NP, int nPPCnt, int StlMaxWallPerBall)
{
	//*--- particle ---*//
	h_Num_Nebor = (int *)malloc(NP * sizeof(int));
	memset(h_Num_Nebor, 0, NP*sizeof(int));  

	h_1DNebor = (int *)malloc(nPPCnt * sizeof(int));
	memset(h_1DNebor, 0, nPPCnt * sizeof(int));


	// number of neighbors
	cutilSafeCall(cudaMalloc((void**)&d_Num_Nebor,     NP * sizeof(int)));
	cudaMalloc((void**)&d_Num_Nebor_Old, NP * sizeof(int));
	cudaMemset(d_Num_Nebor,     0, NP * sizeof(int));
	cudaMemset(d_Num_Nebor_Old, 0, NP * sizeof(int));

	// particle neighbors
	cutilSafeCall(cudaMalloc((void**)&d_Nebor1,     nPPCnt * sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_Nebor1_Old, nPPCnt * sizeof(int)));
	cudaMemset(d_Nebor1,     0, nPPCnt * sizeof(int));
	cudaMemset(d_Nebor1_Old, 0, nPPCnt * sizeof(int));


	cutilSafeCall(cudaMalloc((void**)&d_pMoveforReBld, NP * sizeof(double)));
	cutilSafeCall(cudaMemset(d_pMoveforReBld, 0, NP));
	





	//*--- Stationary wall ---*//
	h_Num_stlWall = (int *)malloc(NP * sizeof(int));
	cudaMalloc((void**)&d_Num_StlWall,     NP * sizeof(int));
	cudaMalloc((void**)&d_Num_StlWall_Old, NP * sizeof(int));

	memset(h_Num_stlWall, 0, NP * sizeof(int));
	cudaMemset(d_Num_StlWall,	  0, NP * sizeof(int));
	cudaMemset(d_Num_StlWall_Old, 0, NP * sizeof(int));

	h_StlWList_1D = (int *)malloc(NP * StlMaxWallPerBall * sizeof(int));
	memset(h_StlWList_1D, 0, NP * StlMaxWallPerBall * sizeof(int));

	cudaMallocPitch(&d_StlW_List,		&pitch_Sw1,		StlMaxWallPerBall * sizeof(int), NP);  
	cudaMallocPitch(&d_StlW_List_Old,	&pitch_Sw2,		StlMaxWallPerBall * sizeof(int), NP); 
	cudaMemset2D(d_StlW_List,			pitch_Sw1, 0,	StlMaxWallPerBall * sizeof(int), NP);
	cudaMemset2D(d_StlW_List_Old,		pitch_Sw2, 0,	StlMaxWallPerBall * sizeof(int), NP);
}








void FeedZoneAlloc(FeedZone *&Feedcube)
{
	Feedcube = (FeedZone *)malloc(sizeof(FeedZone));
}



void ParamAlloc(SIMULATION*& h_dem,   UNIT*& Rdu,    int*& h_lqdCnt, int*& d_lqdCnt, 
	            int*& d_lqdCnt_Old,   MatType*& h_Mat, LIQUID*& h_lqd, 
				PARAMETER*& h_Params, SEARCHCELL &h_cell, SEARCHCELL &d_cell, int nLevels, int NP)
{

	h_dem    = (SIMULATION *)malloc(sizeof(SIMULATION));
	Rdu      = (UNIT *)malloc(sizeof(UNIT));               // Unit conversion
	h_lqdCnt = (int *)malloc(NP * sizeof(int));	           // Liquid contact number
	memset(h_lqdCnt, 0, NP * sizeof(int));	
	
	
	cutilSafeCall(cudaMalloc((void**)&d_lqdCnt, NP * sizeof(int)));     
	cudaMalloc((void**)&d_lqdCnt_Old, NP * sizeof(int)); 

	cudaMemset(d_lqdCnt, 0, NP * sizeof(int));
	cudaMemset(d_lqdCnt_Old, 0, NP*sizeof(int));
	
	
	// cell search
	h_cell.Cellsize = (double* )malloc(nLevels * sizeof(double));
	h_cell.gridSize = (int3* )malloc(nLevels * sizeof(int3));
	h_cell.numCells = (int*)malloc(nLevels * sizeof(int));
	memset(h_cell.Cellsize, 0, nLevels * sizeof(double));
	memset(h_cell.gridSize, 0, nLevels * sizeof(int3));
	memset(h_cell.numCells, 0, nLevels * sizeof(int));

	cutilSafeCall(cudaMalloc((void**)&d_cell.Cellsize, nLevels * sizeof(double)));
	cutilSafeCall(cudaMalloc((void**)&d_cell.gridSize, nLevels * sizeof(int3)));
	cutilSafeCall(cudaMalloc((void**)&d_cell.numCells, nLevels * sizeof(int)));



	//*---- Constant Memory -----*//
	h_Mat    = (MatType *)malloc(NUM_MAT*sizeof(MatType));
	h_lqd    = (LIQUID *)malloc(sizeof(LIQUID));
	h_Params = (PARAMETER *)malloc(sizeof(PARAMETER));
}



void SearchCellAlloc(int NP, int numCells)
{
	// one particle can only belong to one cell/hash
	cudaMalloc((void**)&d_GridParticleHash,  NP * sizeof(int));
	cudaMalloc((void**)&d_GridParticleIndex, NP * sizeof(int));
	cudaMalloc((void**)&d_CellStart,		 numCells * sizeof(int));
	cudaMalloc((void**)&d_CellEnd,			 numCells * sizeof(int));

	cudaMemset(d_GridParticleHash,  0, NP * sizeof(int));        
	cudaMemset(d_GridParticleIndex, 0, NP * sizeof(int));
	cudaMemset(d_CellStart,			0, numCells * sizeof(int));	
	cudaMemset(d_CellEnd,			0, numCells * sizeof(int));
}




void VarInit(double3*& dForce,   double3*& dMom,      double*& dRMom, 
			 int*& d_lqdCnt,     int*& d_lqdCnt_Old,  int NP)
{
	cutilSafeCall(cudaMemcpy(d_lqdCnt_Old, d_lqdCnt, NP * sizeof(int), cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaMemset(d_lqdCnt, 0, NP * sizeof(int)));

	cutilSafeCall(cudaMemset(dForce,   0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dMom,     0, NP * sizeof(double3)));
	cutilSafeCall(cudaMemset(dRMom,    0, NP * sizeof(double)));
}