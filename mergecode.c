#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

#define PI 3.1428571

void weightgen(int i, int span, int Npa, float shift, int npix, int Ndet, int Nslice, float d, float scob, float scd, int *wtno, int *wt)
{
    double y1, y2, x1, x2, z1, z2;
    double dia = 80.86, pixlen = dia / npix;
    double theta = (span * PI) / (Npa * 180);
    int total = 0;

    x1 = scob * sin(theta * i);
    y1 = -scob * cos(theta * i);
    z1 = 51.63 - i * shift / Npa;

    for (int j = 0; j < Nslice; j++) {
        for (int k = 0; k < Ndet; k++) {
            int t = 0;
            double p1 = ((Ndet / 2) - (0.5 + k)) * d;
            double p3 = z1 + ((Nslice / 2) - (j + 0.5)) * d;
            double p2 = scd - scob;
            x2 = p1 * cos(theta * i) - p2 * sin(theta * i);
            y2 = p1 * sin(theta * i) + p2 * cos(theta * i);
            z2 = p3;
            for (int l = 0; l < npix; l++) {
                double y21 = (-dia / 2) + l * pixlen;
                double x21 = x1 + (x2 - x1) * (y21 - y1) / (y2 - y1);
                double z21 = z2 * (y21 - y1) / (y2 - y1);
                if (y21 >= (-dia / 2) && y21 < (dia / 2) && x21 > (-dia / 2) && x21 <= (dia / 2) && z21 > (-dia / 2) && z21 <= (dia / 2)) {
                    int m1 = floor(((dia / 2) - x21) / pixlen);
                    int m2 = floor((y21 + (dia / 2)) / pixlen);
                    int m3 = floor(((dia / 2) - z21) / pixlen);
                    wt[total + t] = m1 * npix * npix + m3 * npix + m2;
                    t++;
                }
            }
            wtno[j * Ndet + k] = t;
            total += t;
        }
    }
}

void calprojdata(int Ndet, int Nslice, float *fini, float *phi, int *wtno, int *wt)
{
    int total1 = 0;
    for (int j = 0; j < Ndet * Nslice; j++) {
        float phibar = 0;
        for (int k = 0; k < wtno[j]; k++) {
            int m = wt[total1 + k];
            phibar += fini[m];
        }
        total1 += wtno[j];
        phi[j] = phibar;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    int npix = 512, Npa = 473, Ndet = 512, Nslice = 512, span = 640;
    float d = 0.8, scob = 166.641, scd = 1238.157, shift = 0.136;
    int Nvox = npix * npix * npix;
    int max_wt_size = (int)(1.8 * npix * Ndet * Nslice);

    int *wtno = (int *)malloc(Ndet * Nslice * sizeof(int));
    int *wt = (int *)malloc(max_wt_size * sizeof(int));
    float *phi = (float *)malloc(Ndet * Nslice * sizeof(float));
    float *fini = (float *)malloc(Nvox * sizeof(float));

    if (!wtno || !wt || !phi || !fini) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("Reading image: %s\n", argv[1]);
    FILE *fid = fopen(argv[1], "r");
    if (!fid) {
        printf("Error opening image file %s\n", argv[1]);
        return 1;
    }

    for (int i = 0; i < npix; i++) {
        for (int j = 0; j < npix; j++) {
            for (int k = 0; k < npix; k++) {
                if (fscanf(fid, "%f", &fini[i * npix * npix + j * npix + k]) != 1) {
                    printf("Error reading image file!\n");
                    return 1;
                }
            }
        }
    }
    fclose(fid);
    printf("Image reading completed.\n");

    long ts = GetTickCount();
    for (int i = 0; i < Npa; i++) {
        printf("Processing projection %d...\n", i + 1);
        printf("Generating weight matrix...\n");
        weightgen(i, span, Npa, shift, npix, Ndet, Nslice, d, scob, scd, wtno, wt);
        printf("Generating number of non-zeros...\n");
        printf("Generating projection data...\n");
        calprojdata(Ndet, Nslice, fini, phi, wtno, wt);
    }
    long te = GetTickCount();
    printf("Total execution time: %ld milliseconds\n", te - ts);

    free(wt);
    free(wtno);
    free(fini);
    free(phi);

    return 0;
}
