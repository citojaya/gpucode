#ifndef HEYI_COMMON_H
#define HEYI_COMMON_H


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include "device_launch_parameters.h"

/*--------------Common definitions---------------*/
#define NUM_MAT 2    // two types material
#define PI  3.1415926
#define gravity 9.8f
#define threadsPerBlock  128
#define IY  index+d_Params.NP
#define IZ  index+2*d_Params.NP
typedef unsigned int uint;



// err detection utility
#define cutilSafeCall(err)   __cudaSafeCall      (err, __FILE__, __LINE__)
#define FPRINTF(a) fprintf a

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		FPRINTF((stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString( err ) ));
		exit(-1);
	}
}


//*--- Boundary Condition ---*// 
struct CylinderBC 
{ 
	double2 cir;     // the position of axial and radius (X, Y)
	double R;        // radius
	double Tw;       // top position
	double Bw;       // bottom position
	double topv, btmv;            
};



struct BoxBC
{
	double xmin, ymin, zmin;
	double xmax, ymax, zmax;
	double topv, btmv;
};






// Particle Feed Zone
struct FeedZone      
{
	double Lmax;
	double Lmin;
};



struct  vGap
{
	double Mn;
	double Mx;
};




// Unit conversion
struct UNIT      
{
	double rlunit, rmunit, rfunit;
	double reunit, rsunit, rtunit, rvunit;  // reduced unit
};



// material properties
struct MatType   
{
	double density;
	double emod, ymod, pois;
	double dmpn;
	double sfrc, rfrc;
	double yldp;        
};





/*----Capillary force/Liquid properties----*/
struct LIQUID        
{
	double stension;     // surface tension 
	double vol;          // liquid volume
	double gapmn;        // Liquid minimum gap
	double layer;     
	double brknmx;       // maximum gap of liquid bridge breakage
};





/*---Simulation Parameters---*/
struct SIMULATION
{
	double dt;           // time step
	double dtFactor;     // reduce time step dtFactor times (0.5~1.0) 0.6
	double cutGap;       // cut gap for neighbor list(0.2~0.5) 0.5
	double ctime;        // run time
	double simTime;      // the total time of simulation
	double fdtime;       // time for particle feed

	double outtime;      // time for output
	double dumptime;     // time for dump

	double outinterval;  // time interval for output 
    double fdinterval;   // time interval for particle feed;
	
	int iter;            // number of time step
	int Outs;            // number of outputs
};






struct LOADMODE
{ 
	int stage;				// 1: Packing    
	double tPacking;		// time for each phase
	int nOutPacking;  

};




struct PPCONTACT_HISTORY
{
	double3 *dispt;  
	double3 *ShrForB;    
	double3 *ShrMomB;    
	double3 *nOld;       

	double *nDsp;        
	double *nDsp_mx;     
	double *plstRad;     
	double *plstDfm;     
	double *bndRad;       
	double *NorForB;     
	double *NorMomB;     
	double *TenStrn;      
	double *ShrStrn;      

	int *stage;       
	int *lqdBrg;         
	int *Exbond;        
};





struct PWCONTACT_HISTORY
{
	double3 *dispt;       

	double  *nDsp;        
	double  *nDsp_mx;     
	double  *plstRad;     
	double  *plstDfm;     

	int    *stage;      
	int    *lqdBrg;      	
};




/*---other common parameters----*/
struct	PARAMETER 
{
	// computing area
	double3 Origin;
	double3 EndPoint;
	double  SearchGap;  // for neighbor find
    double  dt;
	int     NP;         // total particle for simulation
};




struct SEARCHCELL
{
	int*    numCells;   // amount of cells
	int3*   gridSize;   // cell number in each dimension
	double* Cellsize;   // size of cell in millimeter
};




// calculation parameter for p-p contact force
struct calParaPP
{
	double emod;
	double pois;
	double sfrc;
	double ystress;
	double ydsp_t;
	double yforce_t;
	double ystiff_t;
};


// calculation parameter for p-w contact force
struct calParaPW
{
	double emod;
	double pois;
	double sfrc;
	double ystress;
	double ydsp_t;
	double yforce_t;
	double ystiff_t;
};



/*--- Force at contact point---*/
struct FORCEINFO
{
	int     *interact;
	double3 *Fcap;
	double3 *Fn;
	double3 *Ft;
};



/*-- recording force from upward particles --*/
struct pInfoType     
{
	int upCn;        // up contact number
	int cn;          // Contact number
	int lcn;         // Liquid contact number
	double3 upfc;     // Contact force from upward 
	double3 upfl;     // Capillary force from upward
};




struct CONTACTNUMBER
{
	int cnt;
	int cap;
};


struct MutiWallBC
{
	int3 in;                                 // node index
	double vRot;                             // angular velocity value    ################# NOT initialized
	double3 node1, node2, node3;
	double3 Cirk;                            //reference point - inscribing circle
	double3 OrthEdge1, OrthEdge2, OrthEdge3;
	double3 OrthWall;                        // Orthonormal vector to wall
	double3 edge1, edge2, edge3;
	double3 vTra;                            // translational velocity
	double3 rotAxis;                         // normalized vector of the rotational axis
	double3 rotPnt;                          // an arbitrary point on rotational axis
};


struct ZONE
{
	double3 pos; 
	double  porosity;
	int np;          // number of particle
	int idx[80];
};




struct FORCE_LAYER
{
	int nF_cnt;        // # contact force
	int nF_cap;        // # capillary force
	int nF_all;

	double Fcnt;       // contact force
	double Fcap;       // capillary force
	double Fall;       // total force at contact
	
	double cntF_bar;
	double capF_bar;
	double allF_bar;

	double Vel_bar;    // average velocity
	double AngVel_bar;
};


/*-----------------------------------------------*/
/*--------------Global declaration---------------*/ 
/*-----------------------------------------------*/
extern char* genfile;


//*--- particle information ---*//
extern double3 *hPos, *hVel, *hAngVel, *hForce;
extern double *hRad;
extern __device__ double3 *dPos, *dVel, *dAngVel, *dForce, *dMom;   //*dAngPos, 
extern __device__ double  *dRMom, *dRad;









   
//*--- cell search ---*//
extern __device__ int* d_GridParticleHash;                // grid hash value for each particle
extern __device__ int* d_GridParticleIndex;               // particle index for each particle
extern __device__ int* d_CellStart;                       // index of start of each cell in sorted list
extern __device__ int* d_CellEnd;                         // index of end of cell















/*-----------------------------------------------*/
/*----------------CPU functions------------------*/ 
/*-----------------------------------------------*/
extern "C" void ParticleInfoAlloc(int NP);


extern "C" void PPCntHstry_Alloc(PPCONTACT_HISTORY& h_cntHstry_pp, PPCONTACT_HISTORY& d_cntHstry_pp, 
	                             PPCONTACT_HISTORY& d_cntHstry_ppOld, int nPPCnt);

extern "C" void WallCntHstry_Alloc(PWCONTACT_HISTORY& h_cntHstry_Stlpw, PWCONTACT_HISTORY& d_cntHstry_Stlpw, 
	                               PWCONTACT_HISTORY& d_cntHstry_Stlpw_Old, int nWallCnt);

extern "C" void NeborListAlloc(int*& h_Num_Nebor,   int*& h_1DNebor, double*& d_pMoveforReBld, 
	                           int*& d_Num_Nebor,   int*& d_Num_Nebor_Old,
	                           int*& d_Nebor1,      int*& d_Nebor1_Old, 
	                           int*& h_Num_stlWall, int*& h_StlWList_1D, 
	                           int*& d_Num_StlWall, int*& d_Num_StlWall_Old,
	                           int*& d_StlW_List,   int*& d_StlW_List_Old,    
	                           size_t& pitch_Sw1,   size_t& pitch_Sw2,
	                           int NP, int nPPCnt, int StlMaxWallPerBall);



extern "C" void VarInit(double3*& dForce,  double3*& dMom,  double*& dRMom, 
	                    int*& d_lqdCnt,    int*& d_lqdCnt_Old, int NP);





extern "C" void FeedZoneAlloc(FeedZone *&Feedcube);


extern "C" void ParamAlloc(SIMULATION*& h_dem, UNIT*& Rdu, 
	                       int*& h_lqdCnt, int*& d_lqdCnt, int*& d_lqdCnt_Old,
	                       MatType*& h_Mat, LIQUID*& h_lqd, 
	                       PARAMETER *& h_Params, SEARCHCELL &h_cell, SEARCHCELL &d_cell, int nLevels, int NP);


extern "C" void SearchCellAlloc(int NP, int numCells);




extern "C"  void FindRec(FILE *inFile, char* strDest);


extern "C"  void ReadData(double *h_Level_size, int *h_pLevel, int *h_level_nNebors, int *h_level_nPars,int *h_Nebor_Idx,
	                      CylinderBC *&h_Cir, CylinderBC *&d_Cir, FeedZone *Feedcube,
	                      MatType *h_Mat, LIQUID *h_lqd,
	                      PARAMETER *h_Params, double *InitParSize, 
						  int* ForceCap, SIMULATION *h_dem, LOADMODE *h_load, 
						  double *hRad, SEARCHCELL h_cell, int NP, int nLevels, char* genfile, int RESTART);


extern "C" void LoadPara(char *DumpName, int *NP, int *nBatch, int *TotalParticle);



extern "C" void WrtTec(double3 *hPos,		double3 *hVel, 
					   double3 *hForce,		double3 *hAngVel, 
	                   SIMULATION *h_dem,	double *hRad, UNIT *Rdu, int TotalParticle, int NP);


extern "C" void DiaInput(char *diaFile, double *hRad, int NP);



/*----dimensionless conversion----*/
extern "C" void UnitReduce(UNIT *Rdu,  CylinderBC *h_Cir, FeedZone *Feedcube,
	                       LOADMODE *h_load, double *hRad, MatType *h_Mat, 
						   LIQUID *h_lqd, double InitParSize, 
						   SIMULATION *h_dem, PARAMETER *h_Params, SEARCHCELL h_cell, 
						   double *h_Level_size, int nLevels,
						   int NP, int RESTART, char* genfile);


extern "C" int ParticleFeed(FeedZone *Feedcube, CylinderBC *h_Cir, double InitParSize, double3 *hPos, 
	                        double *hRad, int numParticle, int NP);










/*---Constant memory initialization---*/
extern "C" void setMaterial(MatType *h_Mat);
extern "C" void setLiquid(LIQUID *h_lqd);
extern "C" void setParameters(PARAMETER *h_Params);
extern "C" void setCalpara(calParaPP* h_cal, calParaPW* h_pWcal);
extern "C" void IndexOffset(int* d_Num, int *d_nei, int Nn);




/*-----------------------------------------------*/
/*--------------- GPU functions -----------------*/ 
/*-----------------------------------------------*/
extern "C" __global__ void UpdateSystemG(double *dRad, double3 *dPos,  double3 *dVel, double3 *dAngVel, double3 *dForce, 
	                                     double3 *dMom,  double *dRMom,  int* ReNebld,  double* d_pMoveforReBld, int TotalParticle);



extern "C"  __global__ void PPForceCalcG(double3 *dPos, double3 *dVel,    double3 *dAngVel, double *dRad,
	                                     int* d_Nebor1, int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int *d_Nebor_Idx, 
										 PPCONTACT_HISTORY d_cntHstry_pp, 
	                                     int* d_Num_Nebor, int *d_lqdCnt, int *d_lqdCnt_Old, 
	                                     double3 *dForce,  double3 *dMom, double *dRMom,
	                                     int TotalParticle, int NP, int ForceCap, int stage);




extern "C" __global__ void PPCapillaryG(double3 *dPos,  double3 *dForce, double *dRad,
	                                    int *d_nei_Offset, int *d_lqdCnt, int *d_lqdCnt_Old,
	                                    PPCONTACT_HISTORY *d_cntHstry_pp, int* d_Nebor1, size_t pitch1,
										int TotalParticle);




extern "C" __global__ void calcHashG(int*  d_GridParticleHash,                      
	                                 int*  d_GridParticleIndex, int *d_pLevel, SEARCHCELL d_cell, int *d_Ncell_Idx,                    
	                                 double3* dPos, int TotalParticle);               



extern "C" __global__ void reorderDataAndFindCellStartG(int* d_CellStart,          
									  int*   d_CellEnd,            
									  int *  d_GridParticleHash,   
									  int *  d_GridParticleIndex,  
									  int TotalParticle);


extern "C" __global__ void NeighborFindG(double3 *dPos,    int* d_GridParticleIndex, int* d_CellStart, int* d_CellEnd,
	                                     int *d_pLevel,    SEARCHCELL d_cell, int *d_Ncell_Idx, double *d_Level_size,
	                                     int *d_level_nPars, int *d_level_nNebors,	int nLevels,      
										 int* d_Nebor1,     int* d_Nebor_Idx,
	                                     int* d_Num_Nebor, int  TotalParticle, double* dRad);



extern "C" __global__ void HstryCopyG(int* d_Nebor1, int* d_OldNebor1, 
									  int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int* d_Nebor_Idx,
	                                  PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, 
	                                  int *d_Num_Nebor,   int *d_Num_Nebor_Old,
	                                  int TotalParticle, int NP);



extern "C" __global__ void HstryCopyPWG(int* d_StlW_List, int* d_StlW_List_Old, size_t pitch_w1, size_t pitch_w2,
	                                    PWCONTACT_HISTORY d_cntHstry_pw_Old, PWCONTACT_HISTORY d_cntHstry_pw,
	                                    int* d_Num_StlWall, int* d_Num_StlWall_Old,
	                                    int TotalParticle, int NP);



extern "C" __global__ void CirPWInterG(double3 *dPos,  double3 *dVel, double3 *dAngVel, double3 *dForce, double3 *dMom, double *dRMom,
	                                   double *dRad, CylinderBC *d_Cir,   
	                                   PWCONTACT_HISTORY d_cntHstry_Stlpw, int *d_Num_StlWall, int *d_StlW_List, size_t pitch_Sw1, 
	                                   int TotalParticle, int NP, int stage);




extern "C" __global__ void CirWallListG(double3 *dPos, double* dRad, CylinderBC* d_Cir, 
	                                    int* d_StlW_List, size_t pitch_Sw1, int* d_Num_StlWall, 
	                                    int TotalParticle);









/*-----------------------------------------------*/
/*-------------GPU parcel functions--------------*/ 
/*-----------------------------------------------*/
extern "C" void BldNeborList(int* d_GridParticleHash, int* d_GridParticleIndex,
	                         int* d_CellStart, int* d_CellEnd, 
							 int *d_pLevel,    SEARCHCELL d_cell, int *d_Ncell_Idx,double *d_Level_size,
							 int *d_level_nPars, int *d_level_nNebors,	int nLevels, double3 *dPos,         
	                         int Sys_nCells, int *d_Nebor1, int* d_Nebor_Idx, int* d_Num_Nebor, 
	                         int TotalParticle, double* dRad);  



extern "C" void BldCirWall(double3 *dPos, double* dRad, CylinderBC* d_Cir, 
	                       int* d_StlW_List, size_t pitch_Sw1, int* d_Num_StlWall, 
                           int TotalParticle);


extern "C" void HstryCopy(int* d_Nebor1, int* d_OldNebor1, 
						  int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int* d_Nebor_Idx,
	                      PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, 
	                      int *d_Num_Nebor,   int *d_Num_Nebor_Old,
	                      int TotalParticle, int NP);



extern "C" void HstryCopyPW(int* d_StlW_List, int* d_StlW_List_Old, size_t pitch_w1, size_t pitch_w2,
	                        PWCONTACT_HISTORY d_cntHstry_pw_Old, PWCONTACT_HISTORY d_cntHstry_pw,
	                        int* d_Num_StlWall, int* d_Num_StlWall_Old, int TotalParticle, int NP);





extern "C" void PpInterCalc(double3 *dPos, double3 *dVel,    double3 *dAngVel, double *dRad,
	                        int* d_Nebor1, int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int *d_Nebor_Idx,    
							PPCONTACT_HISTORY d_cntHstry_pp, 
	                        int* d_Num_Nebor, int *d_lqdCnt, int *d_lqdCnt_Old, 
	                        double3 *dForce,  double3 *dMom, double *dRMom,
	                        int TotalParticle, int NP, int ForceCap, int stage);





  

extern "C" void CirPWCalc(double3 *dPos,  double3 *dVel, double3 *dAngVel, double3 *dForce, double3 *dMom, double *dRMom,
	                      double *dRad, CylinderBC *d_Cir,   
	                      PWCONTACT_HISTORY d_cntHstry_Stlpw, int *d_Num_StlWall, int *d_StlW_List, size_t pitch_Sw1, 
	                      int TotalParticle, int NP, int stage);






extern "C" void UpdateSystem(double *dRad, double3 *dPos, double3 *dVel, double3 *dAngVel, double3 *dForce, 
							 double3 *dMom, double *dRMom, int* d_ReNebld, double* d_pMoveforReBld, int TotalParticle);



extern "C" void NeborToCPU(int* h_Num_Nebor,        int* d_Num_Nebor,   int* h_1DNebor,     int* d_Nebor1, 
	                       int* h_Num_stlWall,      int* d_Num_StlWall, int* h_StlWList_1D, int* d_StlW_List,
	                       PPCONTACT_HISTORY h_cntHstry_pp,     PPCONTACT_HISTORY d_cntHstry_pp,
	                       PWCONTACT_HISTORY h_cntHstry_Stlpw,  PWCONTACT_HISTORY d_cntHstry_Stlpw, 
	                       int NP, int nPPCnt, size_t pitch_Sw1, int StlMaxWallPerBall);


extern "C" void WallCntHstryCopy(PWCONTACT_HISTORY d_cntHstry_Stlpw_Old, PWCONTACT_HISTORY d_cntHstry_Stlpw, int nWallCnt);
extern "C" void PPCntHstryCopy(PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, int nPPCnt);


#endif 
