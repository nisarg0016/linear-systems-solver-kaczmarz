/*
 * reconstruct.c — Fast CT Reconstruction using SART
 *
 * Pipeline:
 *   1. Read the phantom from sim_512.dat
 *   2. Forward-project (generate sinogram b = Ax)
 *   3. Reconstruct using SART with relaxation + non-negativity
 *   4. Write the reconstruction to a binary file for Python
 *
 * Key fix: when nproj < Npa, projections are evenly spaced across the
 * full angular range to ensure adequate angular coverage (>= 180 deg).
 *
 * Compile (GCC / MinGW):
 *   gcc -O3 -o reconstruct reconstruct.c -lm
 *
 * Usage:
 *   reconstruct.exe <input.dat> <output.bin> [nproj] [sweeps] [relax]
 *
 * Default: 473 projections, 20 sweeps, relaxation=0.5
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PI 3.1428571

/* ================================================================
 * CT geometry (matches mergecode.c)
 * ================================================================ */

static int npix   = 512;
static int Npa    = 473;
static int Ndet   = 512;
static int Nslice = 512;
static int span   = 640;
static double d_  = 0.8;
static double scob = 166.641;
static double scd  = 1238.157;
static double shift_ = 0.136;
static double dia  = 80.86;

/* ================================================================
 * Weight generation (ray tracing for one projection angle)
 *
 * IMPROVED: traces through MAX(|dx|,|dy|) axis planes so that
 * coverage is good for ALL projection angles, not just those
 * with large y-component.  Fixes the 27% voxel coverage gap
 * of the y-plane-only approach.
 *
 * Fills wt[] with voxel indices hit, wtno[] with counts per ray.
 * Returns total number of entries written into wt[].
 * ================================================================ */
static int weightgen(int proj, int *wtno, int *wt)
{
    double pixlen = dia / npix;
    double theta  = (span * PI) / (Npa * 180.0) * proj;
    double cost   = cos(theta), sint = sin(theta);

    double x1 = scob * sint;
    double y1 = -scob * cost;
    double z1 = 51.63 - proj * shift_ / Npa;

    int total = 0;

    for (int j = 0; j < Nslice; j++) {
        for (int k = 0; k < Ndet; k++) {
            int t = 0;
            double p1 = ((Ndet / 2.0) - (0.5 + k)) * d_;
            double p3 = z1 + ((Nslice / 2.0) - (j + 0.5)) * d_;
            double p2 = scd - scob;
            double x2 = p1 * cost - p2 * sint;
            double y2 = p1 * sint + p2 * cost;
            double z2 = p3;

            double dx = x2 - x1;
            double dy = y2 - y1;

            if (fabs(dx) >= fabs(dy)) {
                /* --- Trace through X-planes (ray more in x-direction) --- */
                if (fabs(dx) < 1e-10) { wtno[j * Ndet + k] = 0; total += 0; continue; }
                for (int l = 0; l < npix; l++) {
                    double x21 = (-dia / 2.0) + l * pixlen;
                    double tp  = (x21 - x1) / dx;
                    double y21 = y1 + dy * tp;
                    double z21 = z2 * tp;

                    if (x21 >= -dia / 2.0 && x21 < dia / 2.0 &&
                        y21 >  -dia / 2.0 && y21 <= dia / 2.0 &&
                        z21 >  -dia / 2.0 && z21 <= dia / 2.0) {
                        int m1 = (int)floor((dia / 2.0 - x21) / pixlen);
                        int m2 = (int)floor((y21 + dia / 2.0) / pixlen);
                        int m3 = (int)floor((dia / 2.0 - z21) / pixlen);
                        if (m1 < 0) m1 = 0; if (m1 >= npix) m1 = npix - 1;
                        if (m2 < 0) m2 = 0; if (m2 >= npix) m2 = npix - 1;
                        if (m3 < 0) m3 = 0; if (m3 >= npix) m3 = npix - 1;
                        wt[total + t] = m1 * npix * npix + m3 * npix + m2;
                        t++;
                    }
                }
            } else {
                /* --- Trace through Y-planes (original approach) --- */
                if (fabs(dy) < 1e-10) { wtno[j * Ndet + k] = 0; total += 0; continue; }
                for (int l = 0; l < npix; l++) {
                    double y21 = (-dia / 2.0) + l * pixlen;
                    double tp  = (y21 - y1) / dy;
                    double x21 = x1 + dx * tp;
                    double z21 = z2 * tp;

                    if (y21 >= -dia / 2.0 && y21 < dia / 2.0 &&
                        x21 >  -dia / 2.0 && x21 <= dia / 2.0 &&
                        z21 >  -dia / 2.0 && z21 <= dia / 2.0) {
                        int m1 = (int)floor((dia / 2.0 - x21) / pixlen);
                        int m2 = (int)floor((y21 + dia / 2.0) / pixlen);
                        int m3 = (int)floor((dia / 2.0 - z21) / pixlen);
                        if (m1 < 0) m1 = 0; if (m1 >= npix) m1 = npix - 1;
                        if (m2 < 0) m2 = 0; if (m2 >= npix) m2 = npix - 1;
                        if (m3 < 0) m3 = 0; if (m3 >= npix) m3 = npix - 1;
                        wt[total + t] = m1 * npix * npix + m3 * npix + m2;
                        t++;
                    }
                }
            }
            wtno[j * Ndet + k] = t;
            total += t;
        }
    }
    return total;
}

/* ================================================================
 * Forward-project one angle: phi[ray] = sum of fini[voxels on ray]
 * ================================================================ */
static void forward_project_angle(float *fini, float *phi,
                                  int *wtno, int *wt)
{
    int offset = 0;
    int nrays  = Ndet * Nslice;
    for (int r = 0; r < nrays; r++) {
        double s = 0.0;
        for (int k = 0; k < wtno[r]; k++)
            s += fini[wt[offset + k]];
        phi[r] = (float)s;
        offset += wtno[r];
    }
}

/* ================================================================
 * SART update for one projection angle:
 *   For each ray r:
 *     residual = phi[r] - <a_r, x>
 *     correction[voxels] += residual / weight
 *     count[voxels] += 1
 *   Then: x[v] += relax * correction[v] / count[v]
 *   Then: x[v] = max(x[v], 0)
 * ================================================================ */
static void sart_update(double *x, float *phi,
                        int *wtno, int *wt,
                        double relax,
                        double *correction, int *count)
{
    int nrays  = Ndet * Nslice;
    int nvox   = npix * npix * npix;

    /* Zero accumulators */
    memset(correction, 0, nvox * sizeof(double));
    memset(count, 0, nvox * sizeof(int));

    int offset = 0;
    for (int r = 0; r < nrays; r++) {
        int w = wtno[r];
        if (w == 0) { continue; }

        /* Compute <a_r, x> */
        double dot = 0.0;
        for (int k = 0; k < w; k++)
            dot += x[wt[offset + k]];

        double resid = (double)phi[r] - dot;
        double alpha = resid / (double)w;

        for (int k = 0; k < w; k++) {
            int v = wt[offset + k];
            correction[v] += alpha;
            count[v]++;
        }
        offset += w;
    }

    /* Apply averaged correction + non-negativity */
    for (int v = 0; v < nvox; v++) {
        if (count[v] > 0) {
            x[v] += relax * correction[v] / (double)count[v];
            if (x[v] < 0.0) x[v] = 0.0;
        }
    }
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <input.dat> <output.bin> [nproj] [sweeps] [relax]\n", argv[0]);
        printf("  nproj  : number of projection angles (default 473)\n");
        printf("  sweeps : number of SART sweeps (default 20)\n");
        printf("  relax  : relaxation parameter (default 0.5)\n");
        return 1;
    }

    const char *infile  = argv[1];
    const char *outfile = argv[2];

    /* Disable stdout buffering so redirected output is visible immediately */
    setbuf(stdout, NULL);
    int  nproj   = (argc > 3) ? atoi(argv[3]) : 473;
    int  sweeps  = (argc > 4) ? atoi(argv[4]) : 30;
    double relax = (argc > 5) ? atof(argv[5]) : 1.0;

    if (nproj > Npa) nproj = Npa;

    /* Build evenly-spaced projection index list */
    int *proj_indices = (int *)malloc(nproj * sizeof(int));
    if (!proj_indices) { printf("Alloc failed\n"); return 1; }
    if (nproj == Npa) {
        for (int i = 0; i < nproj; i++) proj_indices[i] = i;
    } else {
        for (int i = 0; i < nproj; i++)
            proj_indices[i] = (int)((double)i * (Npa - 1) / (nproj - 1) + 0.5);
    }
    double theta_first = (span * PI) / (Npa * 180.0) * proj_indices[0];
    double theta_last  = (span * PI) / (Npa * 180.0) * proj_indices[nproj - 1];
    double deg_coverage = (theta_last - theta_first) * 180.0 / PI;

    int nvox  = npix * npix * npix;
    int nrays = Ndet * Nslice;
    int max_wt = (int)(1.8 * npix * nrays);

    printf("=== SART CT Reconstruction ===\n");
    printf("  Volume     : %d^3 = %d voxels\n", npix, nvox);
    printf("  Projections: %d (evenly spaced from %d total)\n", nproj, Npa);
    printf("  Angular coverage: %.1f degrees\n", deg_coverage);
    printf("  Sweeps     : %d\n", sweeps);
    printf("  Relaxation : %.2f\n", relax);
    printf("  Input      : %s\n", infile);
    printf("  Output     : %s\n\n", outfile);

    /* ---- Allocate ---- */
    float  *fini       = (float *)malloc(nvox * sizeof(float));
    double *x          = (double *)calloc(nvox, sizeof(double));
    float  *phi        = (float *)malloc(nrays * sizeof(float));
    float  **sinogram  = (float **)malloc(nproj * sizeof(float *));
    int    *wtno       = (int *)malloc(nrays * sizeof(int));
    int    *wt         = (int *)malloc(max_wt * sizeof(int));
    double *correction = (double *)malloc(nvox * sizeof(double));
    int    *cnt        = (int *)malloc(nvox * sizeof(int));

    if (!fini || !x || !phi || !sinogram || !wtno || !wt || !correction || !cnt) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int p = 0; p < nproj; p++) {
        sinogram[p] = (float *)malloc(nrays * sizeof(float));
        if (!sinogram[p]) { printf("Memory allocation failed for sinogram[%d]!\n", p); return 1; }
    }

    /* ---- Read phantom ---- */
    printf("Reading phantom...\n");
    FILE *fid = fopen(infile, "r");
    if (!fid) { printf("Error: cannot open %s\n", infile); return 1; }
    for (int i = 0; i < nvox; i++) {
        if (fscanf(fid, "%f", &fini[i]) != 1) {
            printf("Error reading value %d\n", i);
            return 1;
        }
    }
    fclose(fid);
    printf("  Phantom loaded: %d voxels\n\n", nvox);

    /* ---- Forward project all angles ---- */
    printf("Forward projection (%d angles)...\n", nproj);
    clock_t t0 = clock();
    for (int p = 0; p < nproj; p++) {
        weightgen(proj_indices[p], wtno, wt);
        forward_project_angle(fini, sinogram[p], wtno, wt);
        if ((p + 1) % 10 == 0 || p == nproj - 1)
            printf("  Projection %d/%d done (angle index %d)\n",
                   p + 1, nproj, proj_indices[p]);
    }
    double fp_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("  Forward projection done in %.1f s\n\n", fp_time);

    /* ---- SART Reconstruction ---- */
    printf("SART reconstruction (%d sweeps)...\n", sweeps);
    clock_t t1 = clock();

    for (int s = 0; s < sweeps; s++) {
        for (int p = 0; p < nproj; p++) {
            /* Re-compute ray geometry for this angle */
            weightgen(proj_indices[p], wtno, wt);
            /* SART update */
            sart_update(x, sinogram[p], wtno, wt, relax, correction, cnt);
        }

        /* Compute RMS error against phantom for this sweep */
        double sse = 0.0;
        double sse_phantom = 0.0;
        for (int v = 0; v < nvox; v++) {
            double diff = x[v] - (double)fini[v];
            sse += diff * diff;
            sse_phantom += (double)fini[v] * (double)fini[v];
        }
        double rel_err = sqrt(sse / (sse_phantom > 0 ? sse_phantom : 1.0));
        double elapsed = (double)(clock() - t1) / CLOCKS_PER_SEC;
        printf("  Sweep %2d/%d  RelErr=%.4f  (%.0fs)\n",
               s + 1, sweeps, rel_err, elapsed);

        /* Save intermediate result every 5 sweeps */
        if ((s + 1) % 5 == 0 || s == sweeps - 1) {
            char tmpname[256];
            snprintf(tmpname, sizeof(tmpname), "recon_sweep%d.bin", s + 1);
            FILE *ftmp = fopen(tmpname, "wb");
            if (ftmp) {
                /* Write clipped copy */
                for (int v = 0; v < nvox; v++) {
                    double val = x[v];
                    if (val < 0.0) val = 0.0;
                    if (val > 1.0) val = 1.0;
                    fwrite(&val, sizeof(double), 1, ftmp);
                }
                fclose(ftmp);
                printf("    -> Saved intermediate: %s\n", tmpname);
            }
        }
    }

    double recon_time = (double)(clock() - t1) / CLOCKS_PER_SEC;
    printf("  Reconstruction done in %.1f s\n\n", recon_time);

    /* ---- Clip to [0, 1] ---- */
    for (int v = 0; v < nvox; v++) {
        if (x[v] < 0.0) x[v] = 0.0;
        if (x[v] > 1.0) x[v] = 1.0;
    }

    /* ---- Save as raw doubles for numpy ---- */
    printf("Saving reconstruction to %s...\n", outfile);
    FILE *fout = fopen(outfile, "wb");
    if (!fout) { printf("Error: cannot open %s for writing\n", outfile); return 1; }
    fwrite(x, sizeof(double), nvox, fout);
    fclose(fout);
    printf("  Saved %d doubles (%.1f MB)\n", nvox, nvox * 8.0 / 1e6);

    /* ---- Cleanup ---- */
    for (int p = 0; p < nproj; p++) free(sinogram[p]);
    free(sinogram); free(fini); free(x); free(phi);
    free(wtno); free(wt); free(correction); free(cnt);
    free(proj_indices);

    printf("\nTotal time: %.1f s\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
    return 0;
}
