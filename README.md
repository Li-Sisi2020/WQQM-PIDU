# WQQM-PIDU-V2.0

WQQM-PIDU model is a water quantity and quality model for paddy irrigation and drainage units, which can simulate water consumption by rice production and wastewater release for a irrigation drainage unit composed of paddy fields, ditches (and a pond if existed). Detail information of the model theory and application examples are referred to a published journal paper Li et al. (2020). For version 2.0, the freshwater irrigation from remote reservoirs outside of irrigation drainage units is restricted in dry conditions beyond the supply capability of the irrigation district. Detailed information is referred to a manuscript Li et al. (2023) submitted to Nature Communications.

The files main_runV2.py, modulesV2.py and sd_functions.py are the core scripts for WQQM-PIDU (V2.0).

The CJ1S-Hubei.py is an example script to run the model, with data_config.json as an example model parameter input file.

The two scripts in folder "WF and ISS analysis" are example post-analysis scripts that can calculate the water footprint (WF) and irrigation self-sufficiency (ISS) of rice-production systems with the modeling results.

## WQQM-PIDU model references:
Li S., Liu H., Zhang L., Li X., Wang H., Zhuang Y., Zhang F., Zhai L., Fan X., Hu W., Pan J., Potential nutrient removal function of naturally existed ditches and ponds in paddy regions: Prospect of enhancing water quality by irrigation and drainage management. Sci. Total Enviro. 718, 137418 (2020)
https://www.sciencedirect.com/science/article/abs/pii/S0048969720309281

Li S., Zhuang Y., Liu H., Wang Z., Zhang F., Lv M., Zhai L., Fan X., Niu S., Chen J., Xu C., Wang N., Ruan S., Shen W., Mi M., Wu S., Du Y., Zhang L., Enhancing rice production sustainability and resilience via reactivating small water bodies for irrigation and drainage. submitted to Nat. Commun. (2023)
