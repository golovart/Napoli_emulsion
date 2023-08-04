#include <float.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "TSystem.h"
#include "TROOT.h"
#include "TMarker.h"
#include "TText.h"
#include "TLatex.h"
#include <vector>
#include <cmath>
#include "TLegend.h"
#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include "/home/scanner-ml/DMDS/dm2root/include/DMRViewHeader.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRAffine2D.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRCluster.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRGrain.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRMicrotrack.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRImage.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRImage.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRImage.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRRun.h"
#include "/home/scanner-ml/DMDS/dm2root/include/DMRRunHeader.h"


using namespace std;

DMRRun  *run=0;
DMRView *v=0;
//gSystem->Load("libDMRoot");



void get_ims( const char *yand_file="yandex_bfcl.txt", const char *file="dm_tracks_cl.dm.root", int dr=16)
{
  run = new DMRRun(file);
  run->GetHeader()->Print();
  run->SetFixEncoderFaults(1);
  v = run->GetView();
  gStyle->SetOptStat("n"); 



	string line, cell;
  int i, i_pol, tr_f, n_pol;
  int ihd, iv, igr;
  int x0,y0,nx,ny;
  int *csv_line = new int[13];
  int *cl_id = new int[8];
  float pixX, pixY;
  int x_ind, y_ind;
  ifstream cl_file (yand_file);
	while ( getline (cl_file,line) )
    {
      istringstream str_line(line);
      i=0;
      while ( getline(str_line, cell, ','))
      {
        stringstream(cell)>>csv_line[i];
        ++i;
      }



    	ihd=csv_line[0]; iv=csv_line[1]; igr=csv_line[2];
      i = 3;
      while (i<11){
        cl_id[i-3]=csv_line[i];
        ++i;
        }
			tr_f=csv_line[11]; n_pol=csv_line[12];
			
			if(n_pol==0){continue;}

			v = run->GetEntry(ihd,1,1,1,1,1,1);

      i_pol=0;
      pixX = run->GetHeader()->pixX;
      pixY = run->GetHeader()->pixY;
      nx   = run->GetHeader()->npixX;
      ny   = run->GetHeader()->npixY;
      DMRViewHeader   *hd = v->GetHD();
      while(cl_id[i_pol]==-1 && i_pol<7){++i_pol;}
      if(hd->flag==0){
        cout<<"\n"<<cl_id[i_pol]<<endl;
    	  DMRCluster      *cl = v->GetCL(cl_id[i_pol]);
      	DMRFrame        *frcl = v->GetFR(cl->ifr);       
      	DMRFrame        *fr  = v->GetFR(frcl->iz,frcl->ipol);
       	cout << "cluster " << hd->aid << " " << hd->flag << " " << cl->igr << " " << cl->ID() << " " << cl->ipol << endl;
     	  x0 = ((cl->x+hd->x)-fr->x)/pixX + nx/2;
  	    y0 = ((cl->y+hd->y)-fr->y)/pixY + ny/2;
        DMRImageCl *im    = run->GetCLIMBFC(ihd,cl_id[i_pol],i_pol,dr,x0,y0);
        TH2F *h = im->GetHist2();

        TString path="";
        path.Form("%s%i%s%i%s%i%s%i%s%i%s%i%s", "csvs/",ihd,"_gr_",igr,"_pol_",i_pol,"_cl_",cl_id[i_pol],"_tr_",tr_f,"_npol_",n_pol,".csv");
        ofstream out_csv(path);
        for(y_ind = h->GetNbinsY(); y_ind >= 2; --y_ind) 
        {
          for(x_ind = 2; x_ind < h->GetNbinsX() + 1; ++x_ind)
          {
            out_csv<< h->GetBinContent(x_ind, y_ind)<<",";
          }
          out_csv<<"\n";
        }
        out_csv.close();
        delete h;
        delete im;

      while(i_pol<7)
      {
        ++i_pol;
        if (cl_id[i_pol]!=-1)
        {
          DMRImageCl *im    = run->GetCLIMBFC(ihd,cl_id[i_pol],i_pol,dr,x0,y0);
          TH2F *h = im->GetHist2();

          TString path="";
          path.Form("%s%i%s%i%s%i%s%i%s%i%s%i%s", "csvs/",ihd,"_gr_",igr,"_pol_",i_pol,"_cl_",cl_id[i_pol],"_tr_",tr_f,"_npol_",n_pol,".csv");
          ofstream out_csv(path);
          for(y_ind = h->GetNbinsY(); y_ind >= 2; --y_ind) 
          {
            for(x_ind = 2; x_ind < h->GetNbinsX() + 1; ++x_ind)
            {
              out_csv<< h->GetBinContent(x_ind, y_ind)<<",";
            }
            out_csv<<"\n";
          }
          out_csv.close();
          delete h;
          delete im;
        }
      }
      //delete hd;
      //delete cl;
      //delete frcl;
      //delete fr;
      }



    }
  cl_file.close();
  delete [] csv_line;
  delete [] cl_id;

}


/*
for future:
make a bash-script to produce images.
remember the order:
> create folder csvs/
> copy get_ims.C
> run: root dm_tracks_cl.dm.root
> gSystem->Load("libDMRoot")
> .x get_ims.C(file, dr)
*/


