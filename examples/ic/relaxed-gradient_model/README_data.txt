
**RELAXED GRADIENT MODEL**

The provided data is organized as follows:
For each investiaged system, data on
	(a) Excited-state geometry optimization (indicated by either "D1-GEO" or "T1-GEO"),
	(b) Derivate coupling calculation at the excited-state geometry (indicated by either "D1-EGRAD" or "T1-EGRAD") and
	(c) Ground-state gradient and Hessian calculation at the excited-state geometry (indicated by "GRAD-HESSIAN")
is provided as separate sub-directory. Directories of type (a) also contain the optimized geometries of the D1 and T1 states, respectively. These geometries are named following a systematic nomenclature:
	"<compound>_<number_unpaired_electrons>_a1_<functional>_<alpha>_<beta>_<gamma>_<basis_set>_opt.xyz",
where <alpha> = 0.190, <beta> = 0.460 and <gamma> = 0.330 refer to the standard parameters of the CAM-B3LYP functional and the def2-SVP basis set is used throughout, 
<basis_set> = svp, and a1 indicates that the geometry refers to the **first excited state**.
