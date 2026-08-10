"""
written by Andreas Koehn (2024, updated 2025)
read info from turbomole projects
- should get a repo of its own
"""

import os

import numpy as np
import re

avmasses = {"h":1.008,"he":4.002602,
"li":6.94,"be":9.0121831,"b":10.81,"c":12.011,"n":14.007,"o":15.999,"f":18.998403163,"ne":20.1797,
"na":22.98976928,"mg":24.305,"al":26.9815385,"si":28.085,"p":30.973761998,"s":32.06,"cl":35.45,"ar":39.948,
"k":39.0983,"ca":40.078,
"sc":44.955908,"ti":47.867,"v":50.9415,"cr":51.9961,"mn":54.938044,
"fe":55.845,"co":58.933194,"ni":58.6934,"cu":63.546,"zn":65.38,
"ga":69.723,"ge":72.630,"as":74.921595,"se":78.971,"br":79.904,"kr":83.798}

atnumbers = {"h":1,"he":2,
        "li":3,"be":4,"b":5,"c":6,"n":7,"o":8,"f":9,"ne":10,
        "na":11,"mg":12,"al":13,"si":14,"p":15,"s":16,"cl":17,"ar":18,
        "k":19,"ca":20,
        "sc":21,"ti":22,"v":23,"cr":24,"mn":25,
        "fe":26,"co":27,"ni":28,"cu":29,"zn":30,
        "ga":31,"ge":32,"as":33,"se":34,"br":35,"kr":36}


class turbomole_results:

    def __init__(self,turbomole_path):
        if not os.path.exists(turbomole_path):
            raise RuntimeError(f"Directory '{turbomole_path}' does not exist")
        if not os.path.isdir(turbomole_path):
            raise RuntimeError(
                f"Provided file ('{turbomole_path}') is not a directory, but for TurboMole a directory was expected"
                )   
        if not os.path.exists(os.path.join(turbomole_path,"control")):
            raise RuntimeError(f"No control file found in {turbomole_path}")
        self.turbomole_path = turbomole_path
        self.control_data = []
        with open(os.path.join(self.turbomole_path,"control")) as control_file:
            self.control_data = control_file.readlines()
        important_datagroups = ["coord","grad"]
        for datagroup in important_datagroups:
            ext_file = self.file_for_datagroup(datagroup)
            if ext_file is None:
                raise RuntimeError(f"No reference found for data group {datagroup}")
            if not os.path.exists(os.path.join(turbomole_path,ext_file)):
                raise RuntimeError(f"No '{ext_file}' file found in {turbomole_path}")
        self.coords = None
        self.masses = None
        self.Lmat = None


    def file_for_datagroup(self,datagroup):
        answer = None
        keyword = "$"+datagroup
        for line in self.control_data:
            if keyword in line:
                if "file" in line:
                    fnamematch = re.search(r"(\w+)\s*=\s*(\w+)",line)
                    if fnamematch is not None:
                        answer = fnamematch.group(2)
                    else:
                        answer = "control"
                else:
                    answer = "control"
                break
        return answer

    def get_coords(self):
        if self.coords is None:
            # read in
            dg_file = self.file_for_datagroup("coord")
            with open(os.path.join(self.turbomole_path,dg_file)) as coord_file:
                data = coord_file.readlines()
                coords = []
                symbols = []
                in_coord = False
                for line in data:
                    if not in_coord:
                        if line.startswith("$coord"):
                            in_coord = True
                            continue
                    if in_coord:
                        if line.startswith("$"):
                            in_coord = False
                            break
                    match = re.search(r"^\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s\s*(\w\w?)",line)
                    if match is None:
                        continue
                    coords.append([float(match.group(1)),float(match.group(2)),float(match.group(3))])
                    symbols.append(match.group(4))
            nAtoms = len(symbols)
            nAtoms2 = len(coords)
            if nAtoms != nAtoms2:
                raise RuntimeError("symbols and coords do not match")
            self.coords = coords
            self.symbols = symbols
            self.nAtoms = nAtoms
        else:
            # get stored values
            coords = self.coords
            symbols = self.symbols
            nAtoms = self.nAtoms

        return coords, symbols, nAtoms


    def get_avmass(self,element):
        return avmasses[element]


    def get_masses(self):

        if self.coords is None:
            self.get_coords()

        if self.masses is None:
            masses = []
            for element in self.symbols:
                masses.append(self.get_avmass(element))
            self.masses = masses
        else:
            masses = self.masses

        return masses


    def get_numbers(self):
        
        if self.coords is None:
            self.get_coords()

        numbers = []
        for element in self.symbols:
            numbers.append(atnumbers[element])

        return numbers


    def get_sqrtMvector(self,flat=False):

        if self.coords is None:
            self.get_coords()

        if self.masses is None:
            self.get_masses()

        sqrtMvec = []
        for idx in range(self.nAtoms):
            sqrtM = np.sqrt(self.masses[idx])
            sqrtMvec.append([sqrtM,sqrtM,sqrtM])

        if flat:
            sqrtMvec = np.reshape(sqrtMvec,(3*self.nAtoms))

        return sqrtMvec


    def get_hessian(self):

        if self.coords is None:
            self.get_coords()
        if self.masses is None:
            self.get_masses()

        ext_file = self.file_for_datagroup("vibrational spectrum")

        # Read frequencies
        with open(os.path.join(self.turbomole_path,ext_file), "r") as vib_spec_file:
            lines = vib_spec_file.readlines()
            assert len(lines) > 1 and lines[0].startswith(
              "$vibrational spectrum"
            ), "vibspectrum file does not have expected format"

            frequencies = np.zeros(3 * self.nAtoms)
            freq_idx = 0
            for current_line in lines[1:]:
                if current_line.strip().startswith("#") or current_line.strip().startswith(
                 "$"
                ):
                    continue

                parts = current_line.split()
                nParts = len(parts)
                assert nParts in [5, 6]

                if nParts == 5:
                    # Translational and rotational DOFs
                    frequencies[freq_idx] = float(parts[1])
                    freq_idx += 1
                elif nParts == 6:
                    # Regular (proper) frequency
                    frequencies[freq_idx] = float(parts[2])
                    freq_idx += 1
        ext_file = self.file_for_datagroup("vibrational normal modes")

        # Read normal modes
        with open(os.path.join(self.turbomole_path, "vib_normal_modes"), "r") as vib_mode_file:
            lines = vib_mode_file.readlines()
            assert len(lines) > 1 and lines[0].startswith(
                "$vibrational normal modes"
            ), "vib_normal_modes file has unexpected format"

            normal_mode_entries = []
            nAtoms = self.nAtoms

            for current_line in lines[1:]:
                if current_line.strip().startswith("#") or current_line.strip().startswith(
                    "$"
                ):
                    continue
                # Ignore element indices at the beginning of the line
                parts = current_line[5:].split()

                normal_mode_entries.extend([float(x) for x in parts])

            # We expect 3N entries per normal mode (x,y,z coordinates for every atom) and in
            # total there should be 3N normal modes, as we are also including the translational
            # and rotational DOFs
            assert len(normal_mode_entries) == 9 * nAtoms * nAtoms

            # Re-assemble the normal coordinates as a 3N x 3N matrix
            Lmat = np.reshape(np.array(normal_mode_entries), (3 * nAtoms, 3 * nAtoms))

            # Re-introduce mass weighting
            for mode in range(3 * nAtoms):
                for row in range(3 * nAtoms):
                    atom: int = row // 3
                    Lmat[row, mode] *= np.sqrt(self.masses[atom])

            # Re-normalize
            for mode in range(3 * nAtoms):
                norm = np.linalg.norm(Lmat[:, mode])
                Lmat[:, mode] /= norm

            # Re-weigthing to get effective masses
            Lmat2 = np.array(Lmat)
            for mode in range(3 * nAtoms):
                for row in range(3 * nAtoms):
                    atom: int = row // 3
                    Lmat2[row, mode] /= np.sqrt(self.masses[atom])            
            
            # some tests (to be removed)
            redmass = 1./np.diag(Lmat2.T @ Lmat2)

            # Safety check
            product = np.matmul(Lmat, Lmat.T)
            assert (
                np.sum(np.abs(product - np.diag(product.diagonal())) > 1e-4) == 0
            ), "Expected Lmat * Lmat^T to be diagonal, but wasn't"

        return frequencies, Lmat, redmass

    def get_gradient(self,index=-1,flat=False):

        ext_file = self.file_for_datagroup("grad")

        with open(os.path.join(self.turbomole_path,ext_file), "r") as grad_file:
            data = grad_file.readlines()

            in_data_group = False
            first_read = True
            grad_read = []
            coord_read = []
            current_grad = []
            current_coord = []
            idx_read = []
            for line in data:
                if "$grad" in line:
                    in_data_group = True
                    continue
                if not in_data_group:
                    continue
                if "cycle" in line:
                    match = re.search(r"cycle\s*=\s*(\d*)",line)
                    if match is not None:
                        idx_read.append(int(match.group(1)))
                    else:
                        raise RuntimeError("struggled to read from line: {line}")
                    # push previous reads
                    if not first_read:
                        grad_read.append(current_grad)
                        coord_read.append(current_coord)
                        current_grad = []
                        current_coord = []
                    else:
                        first_read = False

                    continue

                match = re.search(r"^\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|\d+\.\d*[eEdD][-+]?\d+)\s\s*(\w\w?)",line)
                if match is not None:
                    current_coord.append([float(match.group(1)),float(match.group(2)),float(match.group(3))])
                    continue
                    
                #match = re.search(r"^\s*([-+]?\d*\.\d+|[-+]?\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|[-+]?\d+\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|[-+]?\d+\.\d*[eEdD][-+]?\d+)\s*$",line)
                match = re.search(r"^\s*([-+]?\d*\.\d+|[-+]?\d*\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|[-+]?\d*\.\d*[eEdD][-+]?\d+)\s*([-+]?\d*\.\d+|[-+]?\d*\.\d*[eEdD][-+]?\d+)\s*$",line)
                if match is not None:
                    grd = []
                    for idxgrd in range(1,4):
                        grdstr = match.group(idxgrd).replace("D","e")
                        grd.append(float(grdstr))
                    current_grad.append(grd)
                    continue
            
            # push last read
            if "$" in line:
                in_data_group = False
                grad_read.append(current_grad)
                coord_read.append(current_coord)

        if index > 0:
            for idx,grad_read_idx,coord_read_idx in zip(idx_read,grad_read,coord_read):
                if idx == index:
                    grad = grad_read_idx
                    coord = coord_read_idx
        else:
            grad = grad_read[-1]
            coord = coord_read[-1]

        if flat:
            grad = np.reshape(grad,(3*self.nAtoms))
            coord = np.reshape(coord,(3*self.nAtoms))

        return grad,coord


    def get_couplingvector(self,flat=False):

        if self.coords is None:
            self.get_coords()

        ext_file = self.file_for_datagroup("couplingvector")

        with open(os.path.join(self.turbomole_path,ext_file), "r") as instr:
            data = instr.readlines()
            in_data_group = False
            cvect = []
            count = 0
            for line in data:
                if "$couplingvector" in line:
                    in_data_group = True
                    continue
                if in_data_group:
                    if line[0] == "#":
                        continue
                    if line[0] == "$":
                        in_data_group = False
                        continue
                    line = line.replace("D","e")
                    col = line.split()
                    count +=1
                    cvect.append([float(col[0]),float(col[1]),float(col[2])])
            
            if count % self.nAtoms != 0:
                raise RuntimeError(f"coupling vectors does not match nAtoms: {count} {self.nAtoms}")

        if flat:
            N = int(count/self.nAtoms)
            cvect = np.reshape(cvect,(3*N*self.nAtoms))

        return(cvect)
    
