#include "heydem.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <sm_20_atomic_functions.h> 
#include <math_functions.h>
#include "double_math.h"







// constant variables declaration
__device__ __constant__ struct MatType d_Mat[NUM_MAT];	 // NUM_MAT: number of different material type
__device__ __constant__ struct LIQUID d_lqd;             // constant memory
__device__ __constant__ struct PARAMETER d_Params;
__device__ __constant__ struct calParaPP d_cal;
__device__ __constant__ struct calParaPP d_pWcal;


extern "C" void setMaterial(MatType *h_Mat)
{
	/*---copy material parameters to constant memory---*/
	cudaMemcpyToSymbol(d_Mat, h_Mat, NUM_MAT*sizeof(MatType), 0, cudaMemcpyHostToDevice);
}


extern "C" void setLiquid(LIQUID *h_lqd)
{
	/*---copy liquid properties to constant memory---*/
	cudaMemcpyToSymbol(d_lqd, h_lqd, sizeof(LIQUID), 0, cudaMemcpyHostToDevice);
}



extern "C" void setParameters(PARAMETER *h_Params)
{
	/*--- copy parameters to constant memory--- */
	cudaMemcpyToSymbol(d_Params, h_Params, sizeof(PARAMETER), 0, cudaMemcpyHostToDevice);
}




extern "C" void setCalpara(calParaPP* h_cal, calParaPW* h_pWcal)
{
	cudaMemcpyToSymbol(d_cal, h_cal, sizeof(calParaPP), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pWcal, h_pWcal, sizeof(calParaPW), 0, cudaMemcpyHostToDevice);
}












/*----------- Capillary force ----------*/

__device__ double fCap(double Splus, double ijVol, double effectR)
{
	Splus = max(d_lqd.gapmn, Splus);

	Splus = Splus / pow(ijVol, 0.5);    

	double LogVij = log(ijVol);

	double f1 = -0.44507 - 0.1119* LogVij - 0.012101 * pow(LogVij, 2.0) - 0.0005  * pow(LogVij, 3.0);
	double f2 = 1.9222 -0.0668 * LogVij - 0.0013375 * pow(LogVij, 2.0);
	double f3 = 1.268 + 0.198 * LogVij + 0.02232  * pow(LogVij, 2.0) + 0.0008585 * pow(LogVij, 3.0);
	double f4 = -0.010703 + 0.03345 * LogVij + 0.0018574 * pow(LogVij, 2.0);

	double ForceCap = 2 * PI * effectR * d_lqd.stension * exp(f1 - f2 * exp(f3 * log(Splus) + f4 * pow(log(Splus), 2.0)));

	return ForceCap;
}






//*@@@@@@--- particle-particle contact force ----*/
__device__ void ContactForceD(int index,  int k, int ine,  double nrmDsp, double3 unitVec, 
	                          double iRad,  double3 iVel,  double3 iAngv,
							  double jRad,  double3 jVel,  double3 jAngv,
					          PPCONTACT_HISTORY d_cntHstry_pp, 
							  double3 *deltaFCnt, double3 *deltaMom, double *deltaRmom,
							  int stage, int NP)	

{

	double ijRad = iRad * jRad / (iRad + jRad);

	//*@@@@---- Normal elastic-perfect plastic contact force ----*//
    double fcontactN = 0.0;  

	//*---load, unload or reload verdict---*/
	if(nrmDsp < d_cntHstry_pp.plstDfm[ine])      
		d_cntHstry_pp.stage[ine] = 1;          // no contact
	else if(nrmDsp >= d_cntHstry_pp.plstDfm[ine] && nrmDsp < d_cntHstry_pp.nDsp[ine])
		d_cntHstry_pp.stage[ine] = 3;          // unloading
	else if(nrmDsp >= d_cntHstry_pp.nDsp[ine] && nrmDsp < d_cntHstry_pp.nDsp_mx[ine])
		d_cntHstry_pp.stage[ine] = 4;          // reloading
	else
		d_cntHstry_pp.stage[ine] = 2;          // loading


	if(d_cntHstry_pp.stage[ine] == 1)  
	{
		fcontactN = 0.0;  
	}
	else if(d_cntHstry_pp.stage[ine] == 2)   // loading 
	{
		if(nrmDsp <= d_cal.ydsp_t * ijRad)      
		{
			//*---Hertzian force----*/
			fcontactN = 4.0 / 3.0 * d_cal.emod * pow(ijRad * nrmDsp, 0.5) * nrmDsp;   
			d_cntHstry_pp.plstRad[ine] = ijRad;                                     
			d_cntHstry_pp.plstDfm[ine] = 0.0;       
		}
		else
		{
			//*---plastic force---*/     // (19)
			fcontactN = d_cal.yforce_t * pow(ijRad, 2.0) + d_cal.ystiff_t * ijRad * (nrmDsp - d_cal.ydsp_t * ijRad);       

			//*---harden effect---*/     // (20.21) ----- (29)
			d_cntHstry_pp.plstRad[ine] = ijRad * (4.0/3.0 * d_cal.emod * pow(ijRad * nrmDsp, 0.5)*nrmDsp) / fcontactN;   
			d_cntHstry_pp.plstDfm[ine] = nrmDsp - pow((fcontactN / (4.0/3.0 * d_cal.emod * pow(ijRad, 0.5))), 2.0/3.0);
		}
		d_cntHstry_pp.nDsp_mx[ine] = nrmDsp;

	}
	else
	{
		//*---unload and reload follows a same curve---*/  // (29)
		fcontactN = 4.0/3.0 * d_cal.emod * pow(d_cntHstry_pp.plstRad[ine], 0.5) * pow((nrmDsp - d_cntHstry_pp.plstDfm[ine]), 1.5);
	}
	d_cntHstry_pp.nDsp[ine] = nrmDsp;
	
		
	


	



	



	//*@@@@----- normal damping force -----*/
	double3 ijCPVel = iVel - jVel - cross((iAngv*iRad + jAngv*jRad), unitVec);
	double ijCPVn = dot(ijCPVel, unitVec);
	double fdampingN = 0.0;
	if (stage == 1)
	{
		fdampingN = -2.0 * sqrt(d_Mat[0].dmpn * d_Mat[0].dmpn) * d_cal.emod * sqrt(ijRad * nrmDsp) * ijCPVn;
	}
	

	

	
	///*@@@@----- tangential contact force -----*/
	double3 fcontactTan = make_double3(0.0, 0.0, 0.0);
	double3 ijCPVt = ijCPVel - ijCPVn * unitVec;      
	double3 disptTotal = cross(-1.0 * unitVec, cross(unitVec, d_cntHstry_pp.dispt[ine]));



	if (length(disptTotal) > 1.0e-7)
		disptTotal = disptTotal * length(d_cntHstry_pp.dispt[ine]) / length(disptTotal) + ijCPVt * d_Params.dt; 
	else
		disptTotal = disptTotal + ijCPVt * d_Params.dt; 
	
	

	
	// total maximum tangential displacement
	//double ijDsmx =  d_cal.sfrc * (2.0 - d_cal.pois) / (2.0 - 2.0 * d_cal.pois) * nrmDsp;
	double ijDsmx = 3.0 * d_cal.sfrc * fcontactN / (16.0 * (1-d_cal.pois) * d_cal.emod / (4.0 - 2.0*d_cal.pois) * pow(ijRad*nrmDsp, 0.5));

	double Force_ratio = 1.0;
	if(length(disptTotal) < ijDsmx)
	{
		if (ijDsmx > 1e-7)
			Force_ratio = 1.0 - pow((1.0 - length(disptTotal)/ijDsmx), 1.5);
	}
	else
	{
		Force_ratio = 1.0;
		if (length(disptTotal) > 1.0e-7)
			disptTotal = ijDsmx * disptTotal / length(disptTotal);     // maximum tangential displacement
		else
			disptTotal = ijDsmx * disptTotal;
	}
	d_cntHstry_pp.dispt[ine] = disptTotal;



	if(length(disptTotal)<ijDsmx)
	{
		if(length(disptTotal) > 1.0e-10)
			fcontactTan = -1.0 * d_cal.sfrc * fcontactN * Force_ratio * disptTotal / length(disptTotal);
	}
	else
	{
		if(length(ijCPVt) > 1.0e-10)
			fcontactTan = -1.0 * d_cal.sfrc * fcontactN * Force_ratio * ijCPVt / length(ijCPVt);
	}

	


	//*---Sum force and moments---*//
    *deltaFCnt = (fcontactN + fdampingN) * unitVec + fcontactTan ;
	double gap = iRad - (nrmDsp) / 2.0; 
	*deltaMom  = cross(-gap * unitVec, fcontactTan);
	*deltaRmom = 0.5 * (d_Mat[0].rfrc + d_Mat[0].rfrc) * fcontactN * iRad;

}






/*@@@@@@--- particle-Wall contact force ----*/
__device__ void PWcontactD(int k, double nrmDsp, double3 unitVec, 
	                       double iRad,  double3 iVel,  double3 iAngv,
	                       double3 jVel, PWCONTACT_HISTORY d_cntHstry_Stlpw, 
	                       double3 *deltaFCnt, double3 *deltaMom, double *deltaRmom, 
						   int index, int iwall, int NP, int stage)
{
	//*---definition of reduced material properties---*/
	double ijRad = iRad;
	int ine = k * NP + index;


	//*@@@@---- Normal contact force ----*//
	double fcontactN = 0.0;  
	//*---load, unload or reload verdict---*/  
	if(nrmDsp < d_cntHstry_Stlpw.plstDfm[ine])    	
		d_cntHstry_Stlpw.stage[ine] = 1;         // no contact
	else if(nrmDsp >= d_cntHstry_Stlpw.plstDfm[ine] && nrmDsp < d_cntHstry_Stlpw.nDsp[ine])   
		d_cntHstry_Stlpw.stage[ine] = 3;         // unloading
	else if(nrmDsp >= d_cntHstry_Stlpw.nDsp[ine] && nrmDsp < d_cntHstry_Stlpw.nDsp_mx[ine])
		d_cntHstry_Stlpw.stage[ine] = 4;         // reloading
	else
		d_cntHstry_Stlpw.stage[ine] = 2;         // loading


	if(d_cntHstry_Stlpw.stage[ine]==1)  // no contact-no force 
	{
		fcontactN = 0.0;  
	}
	else if(d_cntHstry_Stlpw.stage[ine] == 2)   // loading 
	{

		if(nrmDsp < (d_pWcal.ydsp_t * ijRad))      
		{
			// Hertzian force
			fcontactN = 4.0 / 3.0 * d_pWcal.emod * pow(ijRad * nrmDsp, 0.5) * nrmDsp;   // ijEmod = 0.549452
			d_cntHstry_Stlpw.plstRad[ine] = ijRad;
			d_cntHstry_Stlpw.plstDfm[ine] = 0.0;
		}
		else
		{
			// plastic force (19)
			fcontactN = (d_pWcal.yforce_t * pow(ijRad, 2.0)) + (d_pWcal.ystiff_t * ijRad) * (nrmDsp - (d_pWcal.ydsp_t * ijRad));  

			// harden effect (20.21)--(29)
			d_cntHstry_Stlpw.plstRad[ine] = ijRad * (4.0/3.0 * d_pWcal.emod * pow(ijRad * nrmDsp, 0.5)*nrmDsp) / fcontactN;   
			d_cntHstry_Stlpw.plstDfm[ine] = nrmDsp - pow((fcontactN / (4.0/3.0 * d_pWcal.emod * pow(ijRad, 0.5))), (2.0/3.0));
		}
		d_cntHstry_Stlpw.nDsp_mx[ine] = nrmDsp;
	}
	else
	{
		// unload and reload follows a same curve (29)
		fcontactN = 4.0/3.0 * d_pWcal.emod * pow(d_cntHstry_Stlpw.plstRad[ine], 0.5) * pow((nrmDsp - d_cntHstry_Stlpw.plstDfm[ine]), 1.5);
	}
	d_cntHstry_Stlpw.nDsp[ine] = nrmDsp;
		
	
	

	//*@@@@----- normal damping force -----*/               
	double3 ijCPVel   = iVel - jVel - cross((iAngv * iRad), unitVec);
	double  ijCPVn    = dot(ijCPVel, unitVec);  
	double  fdampingN = 0.0;
	if (stage == 1 || stage == 7)
	{
		fdampingN = -2.0 * sqrt(d_Mat[0].dmpn * d_Mat[1].dmpn) * d_pWcal.emod * sqrt(ijRad * nrmDsp) * ijCPVn;
	}


	///*@@@@----- tangential contact force -----*/
	double3 ijCPVt = ijCPVel - ijCPVn * unitVec;  
	double3 disptTotal = cross(-1.0 * unitVec, cross(unitVec, d_cntHstry_Stlpw.dispt[ine]));


	if(length(disptTotal) > 1.0e-10)
		disptTotal = disptTotal * length(d_cntHstry_Stlpw.dispt[ine]) / length(disptTotal) + ijCPVt * d_Params.dt; 
	else
		disptTotal = disptTotal + ijCPVt * d_Params.dt; 
	


	// maximum tangential displacement
	double ijDsmx = 3.0 * d_pWcal.sfrc * fcontactN / (16.0 * (1-d_pWcal.pois) * d_pWcal.emod / (4.0 - 2.0*d_pWcal.pois) * pow(ijRad*nrmDsp, 0.5));

	double Force_ratio = 1.0;
	if(length(disptTotal)<ijDsmx)
	{
		if (ijDsmx > 1.0e-10)
			Force_ratio = 1.0 - pow((1.0 - length(disptTotal)/ijDsmx), 1.5);
	}
	else
	{
		Force_ratio = 1.0;
		if (length(disptTotal) > 1.0e-10)
			disptTotal = ijDsmx * disptTotal / length(disptTotal);     
		else
			disptTotal = ijDsmx * disptTotal;

	}
	d_cntHstry_Stlpw.dispt[ine] = disptTotal;



	// implicit zero
	double3 fcontactTan = make_double3(0.0, 0.0, 0.0); 
	if(length(disptTotal)<ijDsmx)
	{
		if(length(disptTotal) > 1.0e-10)
			fcontactTan = -1.0 * d_pWcal.sfrc * fcontactN * Force_ratio * disptTotal / length(disptTotal);
	}
	else
	{
		if(length(ijCPVt) > 1.0e-10)
			fcontactTan = -1.0 * d_pWcal.sfrc * fcontactN * Force_ratio * ijCPVt / length(ijCPVt);
	}


	//*---Sum force and moments---*//
	*deltaFCnt = (fcontactN + fdampingN) * unitVec + fcontactTan ;
	double gap = iRad - (nrmDsp) / 2; 
	*deltaMom  = cross(-gap * unitVec, fcontactTan);
	*deltaRmom =  0.5 * (d_Mat[0].rfrc + d_Mat[0].rfrc) * fcontactN * iRad;

}






/*--------------------------------------------------------*/
/*-------- particle-Wall Interaction Force ---------------*/
/*--------------------------------------------------------*/

__device__ void MultiPWForceD(int index, int k, int iwall,  PWCONTACT_HISTORY d_cntHstry_pw,
	                          double3 iPos, double iRad,    double3 iVel, double3 iAngv,
	                          double3 jPos, double3 jVel,   double3 *dForce, 
							  double3 *dMom, double *dRMom, int NP, int stage)
{
	// i&j information
	double3 ijPos   = iPos - jPos;
	double3 unitVec = ijPos/length(ijPos);         
	double  gap     = length(ijPos) - iRad;


	

	// particle-wall contact force
	if(gap < 0.0)
	{
		double3 deltaFCnt = make_double3(0.0, 0.0, 0.0);
		double3 deltaMom  = make_double3(0.0, 0.0, 0.0);
		double  deltaRmom = 0.0;


		// overlap = -gap
		PWcontactD(k,  -gap, unitVec, iRad, iVel, iAngv,
			       jVel, d_cntHstry_pw, &deltaFCnt, &deltaMom, &deltaRmom,
				   index, iwall, NP, stage);

		//*---idx force and Momentum---*/
		dForce[index] = dForce[index] + deltaFCnt;
		dMom[index]   = dMom[index]   + deltaMom;
		dRMom[index]  = dRMom[index]  + deltaRmom;

	}

}






/*------------------------------------------------------------*/
/*-------------------- Force calculation ---------------------*/
/*------------------------------------------------------------*/
__global__ void PPForceCalcG(double3 *dPos, double3 *dVel,    double3 *dAngVel, double *dRad,
	                         int* d_Nebor1, int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int *d_Nebor_Idx, 
							 PPCONTACT_HISTORY d_cntHstry_pp, int* d_Num_Nebor, int *d_lqdCnt, int *d_lqdCnt_Old, 
							 double3 *dForce,  double3 *dMom, double *dRMom,
							 int TotalParticle, int NP, int ForceCap, int stage)							 
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= TotalParticle) return;  
	

	double3 iPos  = dPos[index];
	double3 iVel  = dVel[index];
	double3 iAngv = dAngVel[index];
	double  iRad  = dRad[index];

	int Inlqd = d_lqdCnt_Old[index];
	if (Inlqd < 12) 
	{ 
		Inlqd = 12; 
	}
	double  iVol = d_lqd.vol * pow((dRad[index] * 2), 3.0) / Inlqd;
	
	
	
	/*------------ Gravity effect --------------- */ 
	dForce[index].z = dForce[index].z - pow(2.0 * dRad[index], 3.0);  
	


	//*-------------------------------------------*//
	//*----------- Force Calculation -------------*//
	//*-------------------------------------------*//
	
	/* size level of the particle */  
	int iLevel = d_pLevel[index];   			                    
	/* the start position in the neighbor list*/
	int nebSId = 0;
	for(int i=0; i<iLevel; i++)
	{
		nebSId = nebSId + d_level_nNebors[i] * d_level_nPars[i];
	}



	for(int k=0; k<d_Num_Nebor[index]; ++k)
	{
		// index of particle j
		int ine = nebSId + k * d_level_nPars[iLevel] + d_Nebor_Idx[index];
		int jdx = d_Nebor1[ine];

		double3 jPos = dPos[jdx];
		double  jRad = dRad[jdx];


        // i&j information  
		double3 ijPos   = iPos  - jPos;
	    double3 unitVec = ijPos / length(ijPos);         
		double  gap = length(ijPos) - iRad - jRad;
	    


		//*-------------------------------------------*/
		//*-------------- contact force --------------*/
		//*-------------------------------------------*/
		if(gap < 0.0)      
		{
			double3 jVel  = dVel[jdx];
			double3 jAngv = dAngVel[jdx];

			double3 deltaFCnt = make_double3(0.0, 0.0, 0.0);
			double3 deltaMom  = make_double3(0.0, 0.0, 0.0);
			double  deltaRmom = 0.0;

			// overlap = -gap
			ContactForceD(index,  k,  ine, -gap, unitVec, 
				          iRad,  iVel,  iAngv,
				          jRad,  jVel,  jAngv, d_cntHstry_pp, 
						  &deltaFCnt, &deltaMom, &deltaRmom,
						  stage, NP);	


			// force and Momentum contribution
			dForce[index] = dForce[index] + deltaFCnt;
			dMom[index]   = dMom[index]   + deltaMom;
			dRMom[index]  = dRMom[index]  + deltaRmom; // Rolling momentum
	
		}
		




		//*-------------------------------------------*/
		//*------------ Capillary Force --------------*/
		//*-------------------------------------------*/
		if (ForceCap == 1)
		{
			int Jnlqd = d_lqdCnt_Old[jdx];
			if (Jnlqd < 12) 
			{ 
				Jnlqd = 12; 
			}


			double jVol = d_lqd.vol * pow((jRad * 2), 3.0) / Jnlqd;
			double ijVol = iVol + jVol;		

			/* rupture distance */
			double Rupture = 0.99 * pow(ijVol, 0.33);  		
			if(gap < 2*d_lqd.layer*(iRad + jRad)) 
				d_cntHstry_pp.lqdBrg[ine] = 1;
			else if (gap > Rupture)
				d_cntHstry_pp.lqdBrg[ine] = 0;


			double Splus = 0.0;

			if(gap < 0){ Splus = 0.0; }
			else { Splus = gap; }
				


			if(d_cntHstry_pp.lqdBrg[ine] == 1)
			{

				double effectR = 2 * iRad * jRad / (iRad + jRad);
				ijVol = ijVol * 8.0;        

				//Splus = Splus / pow(ijVol, 0.5);             ////////////////change 0.5
				//#######################################
				double3 ForceCap = -1.0 * fCap(Splus, ijVol, effectR) * unitVec;

				// force contribution
				dForce[index] = dForce[index] + ForceCap;
				d_lqdCnt[index] = d_lqdCnt[index] + 1;
			}
		}
	}
} 







/*-----------------------------------------------------------*/
/*----------------Update Position & Velocity ----------------*/
/*-----------------------------------------------------------*/
__global__ void UpdateSystemG(double *dRad, double3 *dPos, double3 *dVel, double3 *dAngVel, double3 *dForce, 
	                          double3 *dMom,  double *dRMom, int* d_ReNebld, double* d_pMoveforReBld, int TotalParticle)
{
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= TotalParticle) return;      
    
    double Mass = pow(2.0 * dRad[index], 3.0);		
	double dt = d_Params.dt;
	
    dVel[index].x = dVel[index].x + dForce[index].x * dt / Mass;	// velocity
	dVel[index].y = dVel[index].y + dForce[index].y * dt / Mass;
	dVel[index].z = dVel[index].z + dForce[index].z * dt / Mass;


	dPos[index] = dPos[index] + dVel[index] * dt;                                       // position


	// Accumulated displacement for neighbor list use
	d_pMoveforReBld[index] = d_pMoveforReBld[index] + length(dVel[index]) * d_Params.dt;
	

	// angular velocity
	double Inert = 0.1 * pow(2.0 * dRad[index], 5.0);
	dAngVel[index].x = dAngVel[index].x + dMom[index].x * dt / Inert;
	dAngVel[index].y = dAngVel[index].y + dMom[index].y * dt / Inert;
	dAngVel[index].z = dAngVel[index].z + dMom[index].z * dt / Inert;

	dRMom[index]   = dRMom[index] * dt / Inert;


	double avlength = length(dAngVel[index]);
	if( avlength > 1.0e-7)
	{
		dAngVel[index] = dAngVel[index] * (1.0 - min(1.0, dRMom[index] / avlength));
	}

	// reconstruction of neighbor list
	if (d_pMoveforReBld[index] > (0.3 * d_Params.SearchGap))
	{
		atomicAdd(d_ReNebld, 1);
	}

}





/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/

// calculate position in uniform grid
__device__ int3 calcGridPos(double3 p, int  lvl, SEARCHCELL d_cell)
{
    int3 gridPos;
    // Cellsize: size of the cell, Origin: left bottom coordinate of computational d_Cir
	if (d_cell.Cellsize[lvl] == 0)
	{
		printf("d_Params.Cellsize == 0 .");
	}
	//printf("d_level[%d]_cellsize:%lf.\n", lvl, d_cell.Cellsize[lvl]);

    gridPos.x = max(0, min((int)floor((p.x - d_Params.Origin.x) / d_cell.Cellsize[lvl]), d_cell.gridSize[lvl].x));    //d_Params.Cellsize
    gridPos.y = max(0, min((int)floor((p.y - d_Params.Origin.y) / d_cell.Cellsize[lvl]), d_cell.gridSize[lvl].y));    //d_Params.Cellsize
    gridPos.z = max(0, min((int)floor((p.z - d_Params.Origin.z) / d_cell.Cellsize[lvl]), d_cell.gridSize[lvl].z));    //d_Params.Cellsize
    
    return gridPos;
}




// calculate address in grid from position 
__device__ int calcGridHash(int3 gridPos, int  lvl, SEARCHCELL d_cell)
{
	return (gridPos.z * d_cell.gridSize[lvl].y * d_cell.gridSize[lvl].x) + (gridPos.y * d_cell.gridSize[lvl].x) + gridPos.x;
}



// calculate grid hash value for each particle
__global__ void calcHashG(int* d_GridParticleHash,                     
                          int* d_GridParticleIndex,                      
                          int* d_pLevel, SEARCHCELL d_cell, int *d_Ncell_Idx,
						  double3* dPos, int TotalParticle)              
                        
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >=TotalParticle) return;


    // get address in grid
	int  lvl     = d_pLevel[index];							// particle level
    int3 gridPos = calcGridPos(dPos[index], lvl, d_cell);
    int  hash    = calcGridHash(gridPos,    lvl, d_cell);	// id of cell in current level
	
	int  pre_sum = 0;
	if (lvl != 0)
	{
		pre_sum = d_Ncell_Idx[lvl-1];
	}

	hash = hash + pre_sum;

    // store grid hash and particle index
    d_GridParticleHash[index]  = hash;
    d_GridParticleIndex[index] = index;
    
}






// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartG(int* d_CellStart,          // output: cell start index
				                             int* d_CellEnd,            // output: cell end index
                                             int* d_GridParticleHash,   // input: sorted grid hashes
                                             int* d_GridParticleIndex,  // input: sorted particle indices
					                         int  TotalParticle)  
{
	extern __shared__ int sharedHash[];    
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < TotalParticle)
	{
        hash = d_GridParticleHash[index];
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {
		    sharedHash[0] = d_GridParticleHash[index-1];
	    }
	}


	__syncthreads();
	
	if (index < TotalParticle) 
	{
    	hash = d_GridParticleHash[index];
	    if (index == 0 || hash != sharedHash[threadIdx.x])
	    {
		    d_CellStart[hash] = index;
            if (index > 0)
			    d_CellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == TotalParticle - 1)
        {
            d_CellEnd[hash] = index+1;
        }

	}
      
}




__global__ void NeighborFindG(double3 *dPos,      int* d_GridParticleIndex, int* d_CellStart,   int* d_CellEnd,
							  int *d_pLevel,      SEARCHCELL d_cell,		int *d_Ncell_Idx,	double *d_Level_size,
			                  int *d_level_nPars, int *d_level_nNebors,		int nLevels,		int* d_Nebor1,  int* d_Nebor_Idx,
							  int* d_Num_Nebor,   int  TotalParticle, double* dRad)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= TotalParticle) return;    
  
	double3 Opos = dPos[index];


	/* size level of the particle */  
	int iLevel = d_pLevel[index];   			                    
	/* the start position of memory in the neighbor list*/
	int nebSId = 0;
	for(int i=0; i<iLevel; i++)
	{
		nebSId = nebSId + d_level_nNebors[i] * d_level_nPars[i];
	}

	int numNebor = 0;


    /*--- examine neighboring cells ---*/
	for (int iL=0; iL<nLevels; iL++)
	{
		// get address in the grid of the searched level
		int3 gridPos = calcGridPos(Opos, iL, d_cell);            
		int off_set = (int)ceil(d_Level_size[iLevel] / d_Level_size[iL]);
		
		// Sum of lower level cells
		int  CellSum = 0;
		if (iL != 0)
		{
			CellSum = d_Ncell_Idx[iL-1];
		}


		for(int z=-off_set; z<=off_set; z++) 
		{
			for(int y=-off_set; y<=off_set; y++) 
			{
				// start and end cell of search___current level
				int3 neiPosStart = gridPos + make_int3(-off_set, y, z); 
				int3 neiPosEnd   = gridPos + make_int3(off_set, y, z);

				if (neiPosStart.x < 0) 
				{ 
					neiPosStart.x = 0; 
				}

				if (neiPosEnd.x > (d_cell.gridSize[iL].x - 1)) 
				{ 
					neiPosEnd.x = d_cell.gridSize[iL].x - 1; 
				}

				if (neiPosStart.y < 0 || neiPosStart.z < 0 || neiPosStart.y > d_cell.gridSize[iL].y-1 || neiPosStart.z > d_cell.gridSize[iL].z-1)
				{
					continue;
				}

				
				// Id of particle in start and end cell
				int startIndex = 0xffffffff;
				while(neiPosStart.x != neiPosEnd.x)
				{
					int hash = calcGridHash(neiPosStart, iL, d_cell) + CellSum;	
					startIndex = d_CellStart[hash];
					if (startIndex == 0xffffffff)
						neiPosStart.x = neiPosStart.x + 1;
					else
						break;
				}
				startIndex = d_CellStart[calcGridHash(neiPosStart, iL, d_cell) + CellSum];


				int endIndex = 0xffffffff;
				while(neiPosStart.x != neiPosEnd.x)
				{
					int hash = calcGridHash(neiPosEnd, iL, d_cell) + CellSum;	
					endIndex = d_CellEnd[hash];
					if (endIndex == 0xffffffff)
						neiPosEnd.x = neiPosEnd.x - 1;
					else
						break;
				}
				endIndex = d_CellEnd[calcGridHash(neiPosEnd, iL, d_cell) + CellSum];


				//*--- neighbor check ---*//
				if (startIndex != 0xffffffff)  	
				{
					if(endIndex != 0xffffffff)				
					{
						for(int k=startIndex; k<endIndex; k++) 				
						{
							int RIndex = d_GridParticleIndex[k];     
							if(RIndex != index)
							{
								// check all
								double3 distance = Opos - dPos[RIndex]; 
								
								if(length(distance) < dRad[RIndex] + dRad[index] + d_Params.SearchGap)
								{
									int tId = nebSId + numNebor * d_level_nPars[iLevel] + d_Nebor_Idx[index];

									d_Nebor1[tId] = RIndex;     
									numNebor = numNebor+1;      

									/*if (numNebor > 58)
									{
										printf("index[%d]: %d. \n", index, numNebor);
									}*/
								}
							}
						}
					}
				}
			}
		}
	}



	//*--- Neighbor Number ---*//
	d_Num_Nebor[index] = numNebor;
}












// Contact history copy
__global__ void HstryCopyG(int* d_Nebor1, int* d_OldNebor1, 
						   int *d_pLevel, int *d_level_nNebors, int *d_level_nPars, int* d_Nebor_Idx,
	                       PPCONTACT_HISTORY d_cntHstry_ppOld, PPCONTACT_HISTORY d_cntHstry_pp, 
						   int *d_Num_Nebor,  int *d_Num_Nebor_Old,
						   int TotalParticle, int NP)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= TotalParticle) return;   

	/* size level of the particle */  
	int iLevel = d_pLevel[index];   			                    
	/* the start position in the neighbor list*/
	int nebSId = 0;
	for(int i=0; i<iLevel; i++)
	{
		nebSId = nebSId + d_level_nNebors[i] * d_level_nPars[i];
	}


	for(int i=0; i<d_Num_Nebor[index]; ++i)   
	{
		int ine = nebSId + i * d_level_nPars[iLevel] + d_Nebor_Idx[index];
		int idx = d_Nebor1[ine];

		for (int j=0; j<d_Num_Nebor_Old[index]; ++j)
		{
			int jne = nebSId + j * d_level_nPars[iLevel] + d_Nebor_Idx[index];
			int jdx = d_OldNebor1[jne];

			if (idx == jdx)
			{
				d_cntHstry_pp.dispt[ine]   = d_cntHstry_ppOld.dispt[jne]; 
				d_cntHstry_pp.nDsp[ine]    = d_cntHstry_ppOld.nDsp[jne]; 
				d_cntHstry_pp.nDsp_mx[ine] = d_cntHstry_ppOld.nDsp_mx[jne]; 
				d_cntHstry_pp.plstRad[ine] = d_cntHstry_ppOld.plstRad[jne]; 
				d_cntHstry_pp.plstDfm[ine] = d_cntHstry_ppOld.plstDfm[jne]; 
				d_cntHstry_pp.stage[ine]   = d_cntHstry_ppOld.stage[jne]; 
				d_cntHstry_pp.lqdBrg[ine]  = d_cntHstry_ppOld.lqdBrg[jne]; 
			}
		}
	}
}






__global__ void HstryCopyPWG(int* d_StlW_List, int* d_StlW_List_Old, size_t pitch_w1, size_t pitch_w2,
	                         PWCONTACT_HISTORY d_cntHstry_pw_Old, PWCONTACT_HISTORY d_cntHstry_pw,
							 int* d_Num_StlWall, int* d_Num_StlWall_Old,
	                         int TotalParticle, int NP)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= TotalParticle) return;   


	for(int i=0; i<d_Num_StlWall[index]; ++i)   
	{
		int* row1 = (int*)((char*)d_StlW_List + (index * pitch_w1));
		int idx = row1[i];

		for (int j=0; j<d_Num_StlWall_Old[index]; ++j)
		{
			int* rowOld = (int*)((char*)d_StlW_List_Old + (index * pitch_w2));
			int jdx = rowOld[j];

			if (idx == jdx)
			{
				d_cntHstry_pw.dispt[i * NP + index]   = d_cntHstry_pw_Old.dispt[j * NP + index]; 
				d_cntHstry_pw.nDsp[i * NP + index]    = d_cntHstry_pw_Old.nDsp[j * NP + index]; 
				d_cntHstry_pw.nDsp_mx[i * NP + index] = d_cntHstry_pw_Old.nDsp_mx[j * NP + index]; 
				d_cntHstry_pw.plstRad[i * NP + index] = d_cntHstry_pw_Old.plstRad[j * NP + index]; 
				d_cntHstry_pw.plstDfm[i * NP + index] = d_cntHstry_pw_Old.plstDfm[j * NP + index]; 
				d_cntHstry_pw.stage[i * NP + index]   = d_cntHstry_pw_Old.stage[j * NP + index]; 
				d_cntHstry_pw.lqdBrg[i * NP + index]  = d_cntHstry_pw_Old.lqdBrg[j * NP + index]; 
			}
		}
	}

}






//* ----interaction between particle & wall----- */
__global__ void CirPWInterG(double3 *dPos,  double3 *dVel, double3 *dAngVel, double3 *dForce, double3 *dMom, double *dRMom,
	                        double *dRad, CylinderBC *d_Cir,   
							PWCONTACT_HISTORY d_cntHstry_Stlpw, int *d_Num_StlWall, int *d_StlW_List, size_t pitch_Sw1, 
	                        int TotalParticle, int NP, int stage)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= TotalParticle) return;      

	if (index == 0)  // update wall position
	{
		d_Cir->Tw = d_Cir->Tw + d_Cir->topv * d_Params.dt;
	}



	double3 iPos  = dPos[index];
	double3 iVel  = dVel[index];
	double3 iAngv = dAngVel[index];
	double  iRad  = dRad[index];


	double3 jPos = make_double3(0.0, 0.0, 0.0);
	double3 jVel = make_double3(0.0, 0.0, 0.0);
	
	int *rowi = (int *)((char *)d_StlW_List + (index * pitch_Sw1));

	for(int k=0; k<d_Num_StlWall[index]; ++k)
	{
		// index of wall
		int jw = rowi[k];   

		if (jw == 0)
		{
			jPos = make_double3(iPos.x, iPos.y, d_Cir->Tw);
            jVel = make_double3(0.0, 0.0, d_Cir->topv);

			// particle-wall interaction
			MultiPWForceD(index, k, jw, d_cntHstry_Stlpw,
				          iPos, iRad, iVel, iAngv, jPos, jVel,
						  dForce, dMom, dRMom, NP, stage);
		}
		else if (jw == 1)
		{

			jPos = make_double3(iPos.x, iPos.y, d_Cir->Bw);
			MultiPWForceD(index, k, jw, d_cntHstry_Stlpw,
				          iPos, iRad, iVel, iAngv, jPos, jVel, 
						  dForce, dMom, dRMom, NP, stage);
		}
		else if(jw == 2)
		{
			double DistPCir = pow(pow((d_Cir->cir.x - iPos.x), 2.0) + pow((d_Cir->cir.y - iPos.y), 2.0), 0.5);
			double circosV  = (iPos.x - d_Cir->cir.x) / DistPCir;
			double cirsinV  = (iPos.y - d_Cir->cir.y) / DistPCir;
			jPos.x = d_Cir->R * circosV + d_Cir->cir.x;
			jPos.y = d_Cir->R * cirsinV + d_Cir->cir.y;
			jPos.z = iPos.z;
			
			// particle-wall interaction
			MultiPWForceD(index, k, jw, d_cntHstry_Stlpw,
				          iPos, iRad, iVel, iAngv, jPos, jVel, 
						  dForce, dMom, dRMom, NP, stage);

		}
	}
}










__global__ void CirWallListG(double3 *dPos, double* dRad, CylinderBC* d_Cir, 
	                         int* d_StlW_List, size_t pitch_Sw1, int* d_Num_StlWall, 
	                         int TotalParticle)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= TotalParticle) return; 



	//*--- ip information ---*//
	double3 iPos = dPos[index];
	double  iRad = dRad[index];


	int numW = 0;

	int* rowPW = (int*)((char*)d_StlW_List + (index * pitch_Sw1));

	if (iPos.z >= (d_Cir->Tw - (2.0 * iRad + d_Params.SearchGap)))
	{
		rowPW[numW] = 0;      // top wall 
		numW = numW + 1;
	}

	if (iPos.z <= ((2.0 * iRad + d_Params.SearchGap) + d_Cir->Bw))
	{
		rowPW[numW] = 1;      // bottom wall 
		numW = numW + 1;
	} 

	double DistPCir = pow(pow((d_Cir->cir.x - iPos.x), 2.0) + pow((d_Cir->cir.y - iPos.y), 2.0), 0.5);
	if( DistPCir >= d_Cir->R - (2.0 * iRad + d_Params.SearchGap)) 
	{
		rowPW[numW] = 2;      //  lateral wall
		numW = numW + 1;
	}


	d_Num_StlWall[index] = numW;
}


