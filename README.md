# flash_timer

Extract timing data from FLASH models, and plot scaling performance tests.

At the end of FLASH `.log` files, theres a table that looks like this:

```
perf_summary: code performance summary statistics
                      beginning : 12-14-2020  14:50:31
                         ending : 12-14-2020  14:51:24
   seconds in monitoring period :               52.078
      number of evolution steps :                  100
 ------------------------------------------------------------------------------
 accounting unit                  max/proc (s)  min/proc (s) avg/proc (s)   num calls
 ------------------------------------------------------------------------------
 initialization                          0.385         0.385        0.385           1
  eos                                    0.028         0.028        0.028           5
  guardcell Barrier                      0.001         0.000        0.000           4
  guardcell internal                     0.057         0.057        0.057           4
   amr_guardcell                         0.046         0.046        0.046           4
 evolution                              51.693        51.693       51.693           1
  cosmology                              0.000         0.000        0.000         200
  Hydro                                 47.360        45.462       46.411         100
   Hydro                                47.360        45.462       46.411         100
    eos                                  1.358         1.319        1.339       25600
    guardcell Barrier                    1.879         0.096        0.988         200
[...]
```

This tool extracts this information into a usable table.