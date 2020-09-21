<!-- =============================
     ABOUT
    ============================== -->

\begin{:section, title="About this Package", name="About"}



\end{:section}

<!-- =============================
     GETTING STARTED
     ============================== -->
\begin{:section, title="Walkthrough"}


```julia:ex
using DataFrames, RegressionDiscontinuity, Plots
lee08 = load_rdd_data(:lee08) |> DataFrame
first(lee08, 3)
```

```julia:ex
running_var = RunningVariable(lee08.Zs, cutoff=0.0, treated=:≧);
```

```julia:ex
plot(running_var; ylim=(0,600), bins=40, background_color="#f3f6f9", size=(600,300))
savefig(joinpath(@OUTPUT, "histogram.svg")) # hide
```
\center{\fig{histogram}}


\end{:section}





\begin{:section, title="References"}


**Related Julia packages**
* [GeoRDD.jl](https://github.com/maximerischard/GeoRDD.jl): Package for spatial regression discontinuity designs.
**Related R packages**
* [rdd](https://cran.r-project.org/web/packages/rdd/index.html)
* [optrdd](https://github.com/swager/optrdd)
* [RDHonest](https://github.com/kolesarm/RDHonest)
* [rdrobust](https://cran.r-project.org/web/packages/rdrobust/index.html)


\end{:section}
