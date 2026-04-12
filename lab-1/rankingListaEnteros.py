import sys

nums = [int(a) for a in sys.argv[1:]]
orden_ascendente_unicos = sorted(set(nums))
ranking_por_valor = {v: indice + 1 for indice, v in enumerate(orden_ascendente_unicos)}
print(" ".join(str(ranking_por_valor[n]) for n in nums))
