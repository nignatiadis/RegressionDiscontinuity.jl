using MosekTools
data = load_rdd_data(:lee08)
result = fit(MinMaxOptRD(B=14.28, solver=Mosek.Optimizer), data)
