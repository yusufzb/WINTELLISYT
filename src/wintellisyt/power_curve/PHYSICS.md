# Electrical power reference

Fit the reference on training observations for the intended turbine or study population. Supply wind speed in metres per second as `wind_speed_mps`, measured active electrical power in kilowatts as `power_kw`, and the optional canonical `operational_state`. The calling pipeline owns the site, turbine and time split and maps source channels and state labels to those fields.

The reference groups speeds into half-open 0.5 m/s bins centered on multiples of 0.5 m/s. A speed on a bin boundary enters the higher bin. The first bin has nominal bounds of −0.25 and 0.25 m/s and accepts speeds starting at zero. The prediction uses the arithmetic mean of observed electrical power in the queried bin when the training count meets `min_bin_count`. The default of one defines minimal data support. Studies can require a larger training count and should report the selected threshold and per-bin counts. Sparse bins retain their count and observed mean for inspection. Empty bins report zero observations, and sparse and empty bins return `None` for predictions.

The raw stratum pools every supplied row with finite measurements and nonnegative speed, including signed electrical power values, repeated observations and all operational states. The normal stratum requires an explicit `normal` state from the caller's documented source mapping. A source with missing or unknown state labels can support the raw reference only for those observations. `normal_status` distinguishes a supported normal curve, a normal sample that falls below the bin threshold, and the absence of valid explicitly normal observations. `rejected_count` counts rows with invalid measurements. Source-level duplicate audits and operational-state stratification remain the caller's responsibility.

This research baseline implements the method of bins using the measured wind speed and electrical power directly. IEC power-performance compliance remains unverified. Electrical power reflects turbine control and conversion losses between the rotor and the metering point. Interpreting power differences as faults or estimating recoverable energy would require separate evidence.

The aerodynamic reference for available wind power is 0.5 times air density
times rotor swept area times wind speed cubed. Aerodynamic Cp is rotor
mechanical power divided by that available power. These runs have no verified
air-density correction, rotor-mechanical power measurement or free-stream wind
reference. They therefore estimate no Cp and apply no density normalization.
The publisher nameplates are 12.3 MW across six MM92 turbines at Kelmarsh and
28.7 MW across fourteen MM82 turbines at Penmanshiel, implying 2,050 kW per
turbine from the published aggregate values. The baseline retains measured
electrical values and applies no clipping or rating-based transfer scaling.
The selected SCADA pairs carry no wake classification or sector exclusions.
Wake exposure, curtailment and outages may contribute to observed scatter.
