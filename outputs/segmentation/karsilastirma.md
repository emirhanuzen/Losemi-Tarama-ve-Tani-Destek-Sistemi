# Segmentasyon karşılaştırması

## Yöntem × görüntü tipi

| yontem | tip | goruntu | sinir_gradyani | fisher | kucuk_parca_orani | birlesik_orani | bos_goruntu_orani | nesne_sayisi | sinif_tutarsizligi | sure_ms | ayirma_cizgisi_kontrasti |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kmeans | tümü | 90 | 129.803 | 22.444 | 0.057 | 0.058 | 0.033 | 3 | 0.122 | 16.507 | 1.552 |
| otsu | tümü | 90 | 135.656 | 16.732 | 0.054 | 0.062 | 0.011 | 2.5 | 0.134 | 3.716 | 2.37 |
| watershed | tümü | 90 | 135.487 | 16.7 | 0.053 | 0.058 | 0.011 | 3 | 0.136 | 35.515 | 0.343 |
| kmeans | yakin_cekim | 14 | 98.207 | 59.409 | 0.115 | 0 | 0.071 | 1 | 0.141 | 16.344 | 3.664 |
| otsu | yakin_cekim | 14 | 96.568 | 54.907 | 0.077 | 0 | 0.071 | 1 | 0.179 | 3.859 | 4.009 |
| watershed | yakin_cekim | 14 | 96.568 | 54.907 | 0.077 | 0 | 0.071 | 1 | 0.179 | 38.229 | 2.004 |
| kmeans | yayma | 76 | 135.768 | 20.672 | 0.047 | 0.069 | 0.026 | 3.5 | 0.134 | 16.549 | 1.459 |
| otsu | yayma | 76 | 144.511 | 16.037 | 0.05 | 0.072 | 0 | 3 | 0.161 | 3.668 | 2.364 |
| watershed | yayma | 76 | 144.505 | 15.954 | 0.048 | 0.067 | 0 | 3.5 | 0.163 | 35.006 | 0.343 |

## Sınıf bazında

| yontem | etiket | sinir_gradyani | fisher | kucuk_parca_orani | birlesik_orani | bos_goruntu_orani | nesne_sayisi |
|---|---|---|---|---|---|---|---|
| kmeans | ALL | 141.853 | 24.433 | 0.062 | 0.029 | 0 | 3 |
| kmeans | AML | 115.246 | 20.884 | 0.026 | 0.137 | 0.033 | 17 |
| kmeans | Healthy | 140.721 | 16.759 | 0.085 | 0.009 | 0.067 | 1 |
| otsu | ALL | 142.33 | 18.565 | 0.049 | 0.013 | 0 | 3 |
| otsu | AML | 131.825 | 16.374 | 0.057 | 0.177 | 0.033 | 9.5 |
| otsu | Healthy | 139.431 | 10.18 | 0.057 | 0 | 0 | 1 |
| watershed | ALL | 142.324 | 18.514 | 0.049 | 0.013 | 0 | 3 |
| watershed | AML | 131.836 | 16.519 | 0.052 | 0.164 | 0.033 | 12 |
| watershed | Healthy | 139.431 | 10.054 | 0.057 | 0 | 0 | 1 |

## Seçim sıralaması

| yontem | tip | sinir_gradyani | fisher | kucuk_parca_orani | birlesik_orani | bos_goruntu_orani | sinif_tutarsizligi | ortalama_sira | nihai_skor |
|---|---|---|---|---|---|---|---|---|---|
| kmeans | yayma | 3 | 1 | 1 | 2 | 3 | 1 | 1.83 | 1.75 |
| otsu | yayma | 1 | 2 | 3 | 3 | 1.5 | 2 | 2.08 | 2.12 |
| watershed | yayma | 2 | 3 | 2 | 1 | 1.5 | 3 | 2.08 | 2.12 |
| kmeans | yakin_cekim | 1 | 1 | 3 | 2 | 2 | 1 | 1.67 | 1.75 |
| otsu | yakin_cekim | 2.5 | 2.5 | 1.5 | 2 | 2 | 2.5 | 2.17 | 2.12 |
| watershed | yakin_cekim | 2.5 | 2.5 | 1.5 | 2 | 2 | 2.5 | 2.17 | 2.12 |

Seçilen yöntem: **kmeans**

## K-means k

| yontem | tip | goruntu | sinir_gradyani | fisher | kucuk_parca_orani | birlesik_orani | bos_goruntu_orani | nesne_sayisi | sinif_tutarsizligi | sure_ms | ayirma_cizgisi_kontrasti |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kmeans | tümü | 90 | 129.803 | 22.444 | 0.057 | 0.058 | 0.033 | 3 | 0.122 | 16.507 | 1.552 |
| kmeans_k2 | tümü | 90 | 137.217 | 16.484 | 0.057 | 0.058 | 0.044 | 3 | 0.153 | 14.144 | 1.77 |
| kmeans_k4 | tümü | 90 | 119.88 | 25.772 | 0.063 | 0.065 | 0.022 | 3 | 0.19 | 19.682 | 1.41 |
| kmeans_k5 | tümü | 90 | 107.913 | 27.466 | 0.072 | 0.063 | 0.011 | 3 | 0.284 | 22.722 | 1.259 |
| kmeans | yakin_cekim | 14 | 98.207 | 59.409 | 0.115 | 0 | 0.071 | 1 | 0.141 | 16.344 | 3.664 |
| kmeans_k2 | yakin_cekim | 14 | 98.709 | 52.099 | 0.077 | 0 | 0.071 | 1 | 0.135 | 13.741 | 2.635 |
| kmeans_k4 | yakin_cekim | 14 | 88.823 | 61.267 | 0.115 | 0 | 0.071 | 1 | 0.266 | 18.279 | 2.901 |
| kmeans_k5 | yakin_cekim | 14 | 74.208 | 67.098 | 0.115 | 0 | 0.071 | 1 | 0.294 | 21.269 | 1.849 |
| kmeans | yayma | 76 | 135.768 | 20.672 | 0.047 | 0.069 | 0.026 | 3.5 | 0.134 | 16.549 | 1.459 |
| kmeans_k2 | yayma | 76 | 147.446 | 15.023 | 0.053 | 0.069 | 0.039 | 3 | 0.165 | 14.245 | 1.717 |
| kmeans_k4 | yayma | 76 | 121.39 | 24.579 | 0.054 | 0.076 | 0.013 | 3.5 | 0.183 | 20.109 | 1.334 |
| kmeans_k5 | yayma | 76 | 112.689 | 24.715 | 0.065 | 0.073 | 0 | 5 | 0.264 | 22.895 | 1.249 |
