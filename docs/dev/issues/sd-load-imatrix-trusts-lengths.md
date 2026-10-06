load_imatrix trusts the lengths in the file (out-of-bounds write on a negative name length)

For transparency: AI was used to help analyze and write up this issue. Found while binding the imatrix API in [inferna](https://github.com/shakfu/inferna).

**Environment:** stable-diffusion.cpp `master-898-2bb7294`; still present on `master` (`3f8527a`). Line numbers below are from `master`.

**Evidence:** from source reading; not run against a crafted file.

## Summary

`IMatrixCollector::load_imatrix` (`src/runtime/imatrix.cpp`) reads each entry's name length and allocates from it unchecked (line 267):

```cpp
int len;
in.read((char*)&len, sizeof(len));
std::vector<char> name_as_vec(len + 1);
in.read((char*)name_as_vec.data(), len);
...
name_as_vec[len] = 0;                       // line 274
```

- `len = -1`: the vector is empty and `name_as_vec[-1] = 0` writes before its buffer.
- `len < -1`: `len + 1` converts to a huge `size_t` and the constructor throws through the C API.
- Large `len` or `nval`: allocation is bounded only by the int range.

A failed name read returns `false` without clearing `stats_`, unlike the later failure paths (`stats_ = {}`), so earlier entries of the bad file stay merged into the process-wide matrix.

## Suggested fix

Reject `len <= 0` and `nval <= 0`, cap both (for example by the remaining file size), and clear `stats_` on every failure path, or load into a temporary map and merge on success.
