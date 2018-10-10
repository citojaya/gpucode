
#include <device_launch_parameters.h>
#include "heydem.h"
#include "thrust/device_ptr.h"
#include "thrust/device_free.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/scan.h"




/*-----------------------------------------------------------------------------*/
/*----------------------------extern function----------------------------------*/
/*-----------------------------------------------------------------------------*/

extern "C" 
{

	//Round a / b to nearest higher integer value
	int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}








	/*------------------------------------------------------------------------*/
	/*---invoke kernel for neighbor list building---*/
	/*------------------------------------------------------------------------*/

	void calcHash(int*  d_GridParticleHash, int*  d_GridParticleIndex, 
		          int *d_pLevel, SEARCHCELL d_cell, int *d_Ncell_Idx,
		          double3* dPos, int TotalParticle)
	{   
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 32, numBlocks, numThreads);

		// execute the kernel
		calcHashG<<<numBlocks, numThreads>>>(d_GridParticleHash, d_GridParticleIndex, 
			                                 d_pLevel, d_cell, d_Ncell_Idx, dPos, TotalParticle);
		
	}




	void sortParticles(int *d_GridHash, int *d_GridIndex, int TotalParticle)
	{
		thrust::sort_by_key(thrust::device_ptr<int>(d_GridHash),
		                    thrust::device_ptr<int>(d_GridHash + TotalParticle),
		                    thrust::device_ptr<int>(d_GridIndex));
		


	}


	void IndexOffset(int* d_Num, int *d_nei, int Nn)
	{
		thrust::inclusive_scan(thrust::device_ptr<int>(d_Num), 
	                           thrust::device_ptr<int>(d_Num + Nn), 
		                       thrust::device_ptr<int>(d_nei));
	}


	

	/*------------------------------------------------------------------------*/
	void reorderDataAndFindCellStart(int*  d_CellStart,          // outp
		                             int*  d_CellEnd,            // output						     				 
		                             int*  d_GridParticleHash,   // input: sorted grid hashes
		                             int*  d_GridParticleIndex,  // input: sorted particle indices
		                             int   numCells, int TotalParticle)
	{
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 64, numBlocks, numThreads);

		cutilSafeCall(cudaMemset(d_CellStart, 0xffffffff, numCells*sizeof(int)));
		cutilSafeCall(cudaMemset(d_CellEnd,   0xffffffff, numCells*sizeof(int)));
		int smemSize = sizeof(int)*(numThreads+1);

		// execute the kernel
		reorderDataAndFindCellStartG<<<numBlocks, numThreads, smemSize>>>(d_CellStart,
			                                                              d_CellEnd,
			                                                              d_GridParticleHash,
			                                                              d_GridParticleIndex,
			                                                              TotalParticle);
		
	}






	/*------------------------------------------------------------------------*/
	/*------------------------------------------------------------------------*/
	void NeighborFind(double3* dPos, int* d_GridParticleIndex,     int* d_CellStart, int* d_CellEnd,
		              int *d_pLevel, SEARCHCELL d_cell, int *d_Ncell_Idx, double *d_Level_size, 
					  int *d_level_nPars, int *d_level_nNebors,	int nLevels,
		              int  numCells, int* d_Nebor1, int* d_Nebor_Idx, int* d_Num_Nebor,  
		              int TotalParticle, double* dRad)    
	{
		// thread per particle
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 128, numBlocks, numThreads);

		// execute the kernel
		NeighborFindG<<< numBlocks, numThreads >>>(dPos, d_GridParticleIndex, d_CellStart, d_CellEnd,
			                                       d_pLevel, d_cell, d_Ncell_Idx, d_Level_size, 
												   d_level_nPars, d_level_nNebors,	nLevels,
			                                       d_Nebor1, d_Nebor_Idx, d_Num_Nebor, 
			                                       TotalParticle, dRad);

	}











	void BldNeborList(int* d_GridParticleHash, int* d_GridParticleIndex,
		              int* d_CellStart,        int* d_CellEnd,    
					  int *d_pLevel, SEARCHCELL d_cell, int *d_Ncell_Idx, double *d_Level_size,
					  int *d_level_nPars, int *d_level_nNebors,	int nLevels, double3 *dPos,           
		              int Sys_nCells, int *d_Nebor1,	int* d_Nebor_Idx, int* d_Num_Nebor, 
		              int TotalParticle, double* dRad)   //, double *d_Overlap 

	{
		
		calcHash(d_GridParticleHash, d_GridParticleIndex, 
			     d_pLevel, d_cell, d_Ncell_Idx, dPos, TotalParticle);
		
		sortParticles(d_GridParticleHash, d_GridParticleIndex, TotalParticle);
		
		reorderDataAndFindCellStart(d_CellStart, d_CellEnd, d_GridParticleHash,
			                        d_GridParticleIndex, Sys_nCells, TotalParticle);
	    
		NeighborFind(dPos, d_GridParticleIndex, d_CellStart, d_CellEnd, 
			         d_pLevel,   d_cell, d_Ncell_Idx, d_Level_size, 
					 d_level_nPars, d_level_nNebors,	nLevels,
			         Sys_nCells, d_Nebor1, d_Nebor_Idx, d_Num_Nebor, TotalParticle, dRad);  
		
	}



	



	void HstryCopy(int* d_Nebor1, int* d_OldNebor1, 
				   int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int* d_Nebor_Idx,			
		           PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, 
		           int *d_Num_Nebor,   int *d_Num_Nebor_Old,
		           int TotalParticle, int NP)
	{

		//*---Launch Configuration---*/
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 256, numBlocks, numThreads);

		//*---transfer old information---*/
		HstryCopyG<<<numBlocks, numThreads>>>(d_Nebor1, d_OldNebor1, 
											  d_pLevel, d_level_nNebors, d_level_nPars, d_Nebor_Idx,									
			                                  d_cntHstry_ppOld, d_cntHstry_pp, d_Num_Nebor,  d_Num_Nebor_Old,
			                                  TotalParticle, NP);

	}




	void BldCirWall(double3 *dPos, double* dRad, CylinderBC* d_Cir, 
		            int* d_StlW_List, size_t pitch_Sw1, int* d_Num_StlWall, int TotalParticle)
	{
		//*---Launch Configuration---*/
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 128, numBlocks, numThreads);

		CirWallListG<<<numBlocks, numThreads>>>(dPos, dRad, d_Cir, 
			         d_StlW_List, pitch_Sw1, d_Num_StlWall, 
			         TotalParticle);

	}







	

	void HstryCopyPW(int* d_StlW_List, int* d_StlW_List_Old, size_t pitch_w1, size_t pitch_w2,
		             PWCONTACT_HISTORY d_cntHstry_pw_Old, PWCONTACT_HISTORY d_cntHstry_pw,
		             int* d_Num_StlWall, int* d_Num_StlWall_Old, int TotalParticle, int NP)
	{
		/*---Launch Configuration---*/
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 128, numBlocks, numThreads);

		/*---transfer old information---*/
		HstryCopyPWG<<<numBlocks, numThreads>>>(d_StlW_List, d_StlW_List_Old, pitch_w1, pitch_w2,
			                                    d_cntHstry_pw_Old, d_cntHstry_pw,
			                                    d_Num_StlWall, d_Num_StlWall_Old, TotalParticle, NP);
	}



	// contact & gravity & capillary force
	void PpInterCalc(double3 *dPos, double3 *dVel,    double3 *dAngVel, double *dRad,
		             int* d_Nebor1, int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int *d_Nebor_Idx, 
					 PPCONTACT_HISTORY d_cntHstry_pp, 
		             int* d_Num_Nebor, int *d_lqdCnt, int *d_lqdCnt_Old, 
		             double3 *dForce,  double3 *dMom, double *dRMom,
		             int TotalParticle, int NP, int ForceCap, int stage)
	{
		/*---Launch Configuration---*/
		cudaFuncSetCacheConfig(PPForceCalcG, cudaFuncCachePreferL1);
		int n_blocks = TotalParticle/threadsPerBlock + (TotalParticle % threadsPerBlock == 0? 0:1);	
		
		/*--- contact force & gravity & capillary force ---*/
		PPForceCalcG<<<n_blocks, threadsPerBlock>>>(dPos,  dVel,   dAngVel, dRad, d_Nebor1, 
													d_pLevel, d_level_nNebors, d_level_nPars, d_Nebor_Idx,
			                                        d_cntHstry_pp, d_Num_Nebor,   d_lqdCnt, d_lqdCnt_Old,
			                                        dForce,  dMom, dRMom, 
			                                        TotalParticle, NP, ForceCap, stage);

		cutilSafeCall(cudaGetLastError());
	}





	void CirPWCalc(double3 *dPos,  double3 *dVel, double3 *dAngVel, double3 *dForce, double3 *dMom, double *dRMom,
		           double *dRad, CylinderBC *d_Cir,   
		           PWCONTACT_HISTORY d_cntHstry_Stlpw, int *d_Num_StlWall, int *d_StlW_List, size_t pitch_Sw1, 
		           int TotalParticle, int NP, int stage)
	{
		/*---Launch Configuration---*/
		int n_blocks = TotalParticle/threadsPerBlock + (TotalParticle % threadsPerBlock == 0? 0:1);	
		cudaFuncSetCacheConfig(CirPWInterG, cudaFuncCachePreferL1);
		
		//*--- cylinder ---*/
		CirPWInterG<<<n_blocks, threadsPerBlock>>>(dPos, dVel, dAngVel, dForce, dMom, dRMom, dRad, d_Cir,
			                                       d_cntHstry_Stlpw, d_Num_StlWall, 
												   d_StlW_List, pitch_Sw1, TotalParticle, NP, stage);

		
	}






	// update and integration
	void UpdateSystem(double *dRad, double3 *dPos,  double3 *dVel,  double3 *dAngVel, double3 *dForce, // double3 *dAngPos,
		              double3 *dMom,  double *dRMom,  int* d_ReNebld, double* d_pMoveforReBld, int TotalParticle)
	{
		/*---Launch Configuration---*/
		int numThreads, numBlocks;
		computeGridSize(TotalParticle, 256, numBlocks, numThreads);
		cudaFuncSetCacheConfig(UpdateSystemG, cudaFuncCachePreferL1);

		UpdateSystemG<<<numBlocks, numThreads>>>(dRad, dPos, dVel, dAngVel, dForce, dMom, dRMom, 
			                                     d_ReNebld, d_pMoveforReBld, TotalParticle);
	}





	// Copy Neighbor list and Contact history to CPU
	void NeborToCPU(int* h_Num_Nebor,       int* d_Num_Nebor,   int* h_1DNebor,     int* d_Nebor1, 
		            int* h_Num_stlWall,     int* d_Num_StlWall, int* h_StlWList_1D, int* d_StlW_List,
		            PPCONTACT_HISTORY h_cntHstry_pp,    PPCONTACT_HISTORY d_cntHstry_pp,
		            PWCONTACT_HISTORY h_cntHstry_Stlpw, PWCONTACT_HISTORY d_cntHstry_Stlpw, 
		            int NP, int nPPCnt, size_t pitch_Sw1, int StlMaxWallPerBall)
	{
		
		//*--- particle ---*/
		cudaMemcpy(h_Num_Nebor, d_Num_Nebor, NP * sizeof(int),   cudaMemcpyDeviceToHost);
		cudaMemcpy(h_1DNebor,   d_Nebor1,    nPPCnt*sizeof(int), cudaMemcpyDeviceToHost);
		
		// particle contact history
		cudaMemcpy(h_cntHstry_pp.dispt,   d_cntHstry_pp.dispt,   nPPCnt*sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.nDsp,    d_cntHstry_pp.nDsp,    nPPCnt*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.nDsp_mx, d_cntHstry_pp.nDsp_mx, nPPCnt*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.plstRad, d_cntHstry_pp.plstRad, nPPCnt*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.plstDfm, d_cntHstry_pp.plstDfm, nPPCnt*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.stage,   d_cntHstry_pp.stage,   nPPCnt*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_pp.lqdBrg,  d_cntHstry_pp.lqdBrg,  nPPCnt*sizeof(int), cudaMemcpyDeviceToHost);
		
		  
		//*--- stationary wall ---*/
		cudaMemcpy(h_Num_stlWall,      d_Num_StlWall,      NP * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy2D(h_StlWList_1D,   StlMaxWallPerBall*sizeof(int), d_StlW_List, pitch_Sw1, StlMaxWallPerBall*sizeof(int), NP, cudaMemcpyDeviceToHost);

		// particle contact history
		int nWallCnt = NP * StlMaxWallPerBall;
		cudaMemcpy(h_cntHstry_Stlpw.dispt,   d_cntHstry_Stlpw.dispt,   nWallCnt*sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.nDsp,    d_cntHstry_Stlpw.nDsp,    nWallCnt*sizeof(double),  cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.nDsp_mx, d_cntHstry_Stlpw.nDsp_mx, nWallCnt*sizeof(double),  cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.plstRad, d_cntHstry_Stlpw.plstRad, nWallCnt*sizeof(double),  cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.plstDfm, d_cntHstry_Stlpw.plstDfm, nWallCnt*sizeof(double),  cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.stage,   d_cntHstry_Stlpw.stage,   nWallCnt*sizeof(int),     cudaMemcpyDeviceToHost);
		cudaMemcpy(h_cntHstry_Stlpw.lqdBrg,  d_cntHstry_Stlpw.lqdBrg,  nWallCnt*sizeof(int),     cudaMemcpyDeviceToHost);
		
	

		
	}

	void WallCntHstryCopy(PWCONTACT_HISTORY d_cntHstry_Stlpw_Old, PWCONTACT_HISTORY d_cntHstry_Stlpw, int nWallCnt)
	{
		cudaMemcpy(d_cntHstry_Stlpw_Old.dispt,   d_cntHstry_Stlpw.dispt,   nWallCnt * sizeof(double3), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_Stlpw_Old.nDsp,    d_cntHstry_Stlpw.nDsp,    nWallCnt * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_Stlpw_Old.nDsp_mx, d_cntHstry_Stlpw.nDsp_mx, nWallCnt * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_Stlpw_Old.plstRad, d_cntHstry_Stlpw.plstRad, nWallCnt * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_Stlpw_Old.plstDfm, d_cntHstry_Stlpw.plstDfm, nWallCnt * sizeof(double), cudaMemcpyDeviceToDevice);
		
		cudaMemcpy(d_cntHstry_Stlpw_Old.stage,   d_cntHstry_Stlpw.stage,   nWallCnt * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_Stlpw_Old.lqdBrg,  d_cntHstry_Stlpw.lqdBrg,  nWallCnt * sizeof(int), cudaMemcpyDeviceToDevice);

		cudaMemset(d_cntHstry_Stlpw.dispt,   0, nWallCnt * sizeof(double3));
		cudaMemset(d_cntHstry_Stlpw.nDsp,    0, nWallCnt * sizeof(double));
		cudaMemset(d_cntHstry_Stlpw.nDsp_mx, 0, nWallCnt * sizeof(double));
		cudaMemset(d_cntHstry_Stlpw.plstRad, 0, nWallCnt * sizeof(double));
		cudaMemset(d_cntHstry_Stlpw.plstDfm, 0, nWallCnt * sizeof(double));
		cudaMemset(d_cntHstry_Stlpw.stage,   0, nWallCnt * sizeof(int));
		cudaMemset(d_cntHstry_Stlpw.lqdBrg,  0, nWallCnt * sizeof(int));
	}



	void PPCntHstryCopy(PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, int nPPCnt)
	{
		cudaMemcpy(d_cntHstry_ppOld.dispt,   d_cntHstry_pp.dispt,   nPPCnt * sizeof(double3), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.nDsp,    d_cntHstry_pp.nDsp,    nPPCnt * sizeof(double),  cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.nDsp_mx, d_cntHstry_pp.nDsp_mx, nPPCnt * sizeof(double),  cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.plstRad, d_cntHstry_pp.plstRad, nPPCnt * sizeof(double),  cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.plstDfm, d_cntHstry_pp.plstDfm, nPPCnt * sizeof(double),  cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.stage,   d_cntHstry_pp.stage,   nPPCnt * sizeof(int),     cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cntHstry_ppOld.lqdBrg,  d_cntHstry_pp.lqdBrg,  nPPCnt * sizeof(int),     cudaMemcpyDeviceToDevice);
		
		cudaMemset(d_cntHstry_pp.dispt,   0, nPPCnt * sizeof(double3));
		cudaMemset(d_cntHstry_pp.nDsp,    0, nPPCnt * sizeof(double));
		cudaMemset(d_cntHstry_pp.nDsp_mx, 0, nPPCnt * sizeof(double));
		cudaMemset(d_cntHstry_pp.plstRad, 0, nPPCnt * sizeof(double));
		cudaMemset(d_cntHstry_pp.plstDfm, 0, nPPCnt * sizeof(double));
		cudaMemset(d_cntHstry_pp.stage,   0, nPPCnt * sizeof(int));
		cudaMemset(d_cntHstry_pp.lqdBrg,  0, nPPCnt * sizeof(int));
	}




	

}  //extern "C" 