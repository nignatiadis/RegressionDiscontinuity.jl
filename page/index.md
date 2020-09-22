
<!-- =============================
     GETTING STARTED
     ============================== -->
\begin{:section, title="Walkthrough"}
 
Let us load the dataset from [Lee (2018)](#ref-lee2018). We will reproduce analyses from [Imbens and Kalyanaraman (2012)](#ref-ik2012).

```julia:ex
using DataFrames, RegressionDiscontinuity, Plots
ENV["GKSwstype"] = "nul"  #hide
lee08 = load_rdd_data(:lee08) |> DataFrame
first(lee08, 3)
```

```julia:ex
running_var = RunningVariable(lee08.Zs, cutoff=0.0, treated=:≧);
```

Let us first plot the histogram of the running variable:
```julia:ex
plot(running_var; ylim=(0,600), bins=40, background_color="#f3f6f9", size=(700,400))
savefig(joinpath(@OUTPUT, "histogram.svg")) # hide
```
\center{\fig{histogram}}

Next we plot the regressogram (also known as scatterbin) of the response:

```julia:ex
regressogram = plot(running_var, lee08.Ys; bins=40, background_color="#f3f6f9", size=(700,400), legend=:bottomright)
savefig(regressogram, joinpath(@OUTPUT, "regressogram.svg")) # hide
```
\center{\fig{regressogram}}
 
 We observe a jump at the discontinuity, which we can estimate, e.g., with local linear regression. We use local linear regression with rectangular kernel and choose bandwidth with the Imbens-Kalyanaraman bandwidth selector:
 
 ```julia:ex
rect_ll_rd = fit(NaiveLocalLinearRD(kernel=Rectangular(), bandwidth=ImbensKalyanaraman()),
                 running_var, lee08.Ys)
 ```
 
 ```julia:ex
 plot!(regressogram, rect_ll_rd; show_local_support=true)
 savefig(joinpath(@OUTPUT, "regressogram_plus_llfit.svg")) # hide
 ```
 \center{\fig{regressogram_plus_llfit}}
 
 Let zoom in on the support of the local kernel and also with more refined regerssogram:
 
 ```julia:ex
local_regressogram = plot(rect_ll_rd.data_subset; bins=40, background_color="#f3f6f9", size=(700,400), legend=:bottomright)
plot!(rect_ll_rd)
savefig(joinpath(@OUTPUT, "local_regressogram.svg")) # hide
```
 \center{\fig{local_regressogram}}
 
 
 Finally, We could repeat all of the above analysis with another kernel, e.g. the triangular kernel.
 
 ```julia:ex
triang_ll_rd = fit(NaiveLocalLinearRD(kernel=SymTriangularDist(), bandwidth=ImbensKalyanaraman()),
				   running_var, lee08.Ys)
 ```

\end{:section}





\begin{:section, title="References"}

**Publications**
* \label{ref-ik2012} Imbens, Guido, and Karthik Kalyanaraman. "Optimal bandwidth choice for the regression discontinuity estimator." The Review of economic studies 79.3 (2012): 933-959.
* \label{ref-lee2018} Lee, David S. "Randomized experiments from non-random selection in US House elections." Journal of Econometrics 142.2 (2008): 675-697.

**Related Julia packages**
* [GeoRDD.jl](https://github.com/maximerischard/GeoRDD.jl): Package for spatial regression discontinuity designs.
**Related R packages**
* [rdd](https://cran.r-project.org/web/packages/rdd/index.html)
* [optrdd](https://github.com/swager/optrdd)
* [RDHonest](https://github.com/kolesarm/RDHonest)
* [rdrobust](https://cran.r-project.org/web/packages/rdrobust/index.html)





\end{:section}
