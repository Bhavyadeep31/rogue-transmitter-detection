# LRP Topic 1: SDR-Based Spectrum Monitoring Architectures for 3GPP N78

**Topic wording:** SDR-based spectrum monitoring architectures for 3GPP N78 (3.3–3.8 GHz): hardware, firmware, sampling and capture chains.

**Research focus:** a receive-only, student-scale architecture that can detect and record RF activity in selected N78 sub-bands and provide well-described IQ captures for later spectrum sensing. This is an initial literature and design note, not a claim of production-grade cellular monitoring.

## 1. Topic in one paragraph

3GPP NR band n78 is a TDD operating band spanning 3300–3800 MHz. The 500 MHz span is an allocation range, not a single NR carrier to capture in one shot. 3GPP channel bandwidths in FR1 are at most 100 MHz, and n78 deployment configurations vary. A monitor therefore needs either (a) a receiver with enough instantaneous bandwidth for the carrier/sub-band of interest, (b) a tunable receiver that scans sub-bands over time, or (c) several frequency-coherent receiver channels whose combined passbands cover more spectrum. Even the multi-channel option does not automatically provide uninterrupted 500 MHz coverage: the tuning range, usable analog passband, host link, storage and channel synchronization all have to support it.

## 2. End-to-end architecture

```text
N78 antenna
   ↓
3.3–3.8 GHz preselection / attenuation / optional LNA
   ↓
SDR RF front end: tune, gain, analog channel filter, quadrature downconversion
   ↓
I/Q ADCs → FPGA: digital downconversion, decimation, timestamping, buffering
   ↓ USB 3 / PCIe / 10 GbE (device dependent)
Host UHD driver → capture process → IQ file + metadata + health/overflow log
   ↓
Offline spectrum/occupancy analysis and later detector pipeline
```

The RF front end sets what reaches the ADC; the FPGA and host interface determine how much of that data can be delivered continuously. The capture chain should preserve sample-rate and time information, and surface every discontinuity rather than silently treating a damaged capture as contiguous.

### Functional blocks to discuss in the literature review

1. **Antenna and RF conditioning:** antenna, cable loss, optional band-pass preselector, attenuator, and only then a low-noise amplifier if the link budget needs it. At 3.5 GHz, strong nearby signals outside the target carrier can compress the receiver or create intermodulation; the exact need for filtering depends on the site. Gain must be selected to avoid both a noise-limited capture and ADC/front-end clipping.
2. **Tunable receiver:** center-frequency LO, RF gain, analog low-pass/channel filter and I/Q conversion. A wide tuning range does not imply good sensitivity or flat response across the range; hardware performance should be characterized near the actual 3.3–3.8 GHz operating point.
3. **Digital front end:** the ADC produces complex I/Q samples after quadrature conversion; FPGA DDC/filters can shift and decimate to the application sample rate. Choose a rate that covers the wanted signal plus transition/guard margin, not just the nominal occupied resource blocks.
4. **Streaming and capture:** device buffers, transport link, host buffers and disk form one throughput budget. Record the device timestamp, center frequency, sample rate, gain/AGC mode, bandwidth/filter, channel, antenna, clock source, format, software/FPGA versions and overflow events with each capture.
5. **Analysis interface:** saved IQ feeds PSD/waterfall, occupancy, candidate detection, and later 5G NR-aware processing. The acquisition layer should be reusable independently of the detector.

## 3. Hardware options and tradeoffs

| Example class | Relevant published capability | What it can support in this project | Main caveat |
|---|---|---|---|
| **USRP B210 / AD9361 class** | Tuning coverage 70 MHz–6 GHz; up to 56 MHz instantaneous RF bandwidth and 61.44 MS/s quadrature streaming | Lowest-complexity prototype for one N78 carrier/sub-band, scanning by retuning; two coherent receive paths are available | It cannot observe the full 500 MHz n78 range at once. USB/host throughput and performance at the chosen 3.5 GHz point need validation. |
| **USRP X310 class** | Up to 160 MHz usable bandwidth per channel (daughterboard dependent); 200 MS/s processing; high-throughput host options | Wider contiguous slice, more FPGA/transport headroom, possible multi-channel experiments | Daughterboards are separate and materially affect RF performance; still less than 500 MHz on one channel. |
| **USRP N310 class** | Four RX channels, each up to 100 MHz instantaneous bandwidth; 10 MHz–6 GHz range; 10 GbE and shared timing options | Advanced parallel sub-band monitoring or spatial measurement experiments | Expensive/complex for the MVP. Four 100 MHz channels do not guarantee gap-free 500 MHz coverage; overlapping tuned passbands, clock/LO synchronization and aggregate transport must be designed. |

These are representative architectures, not a purchase recommendation. The B210 is a reasonable MVP baseline if available, with a scanning strategy or fixed-carrier scope. The X310/N310 are references for how wider instantaneous bandwidth and channel count scale, not prerequisites for the literature topic.

## 4. Sampling, bandwidth, and storage calculations

For complex baseband samples, the sample rate is the complex-I/Q rate: an ideal rate of 50 MS/s represents approximately 50 MHz of complex bandwidth around the tuned center, subject to real RF/analog filter roll-off and practical guard margins. This differs from real-only sampling, where the Nyquist relation is often described as twice the bandwidth.

An NR carrier can be narrower than the whole n78 allocation. In current 3GPP TS 38.104 tables, FR1 supports channel bandwidth configurations up to 100 MHz, with 30 kHz SCS including 100 MHz; 15 kHz SCS configurations top out at 50 MHz in that table. The actual carrier's bandwidth/SCS must be known before setting the capture rate. A B210's maximum 56 MHz instantaneous bandwidth is consequently suitable for narrower carriers but not a complete 100 MHz carrier capture.

For `sc16` (16-bit I plus 16-bit Q), uncompressed storage is:

`bytes/s = complex_samples/s × 4`

| Complex sample rate | Approx. RF view | `sc16` rate | Storage per minute |
|---:|---:|---:|---:|
| 10 MS/s | narrow sub-band / prototype | 40 MB/s | 2.4 GB |
| 30.72 MS/s | ~30 MHz view | 122.9 MB/s | 7.37 GB |
| 50 MS/s | ~50 MHz view | 200 MB/s | 12 GB |
| 61.44 MS/s | B210 maximum stream rate | 245.8 MB/s | 14.75 GB |

These are decimal units and exclude file-system, metadata, and any second channel overhead. `sc8` halves the wire sample bytes versus `sc16`, easing host-link pressure but reduces amplitude resolution and increases quantization noise. For an MVP, capture short bursts at a stable 10–30.72 MS/s first, then measure the complete path before increasing the rate.

### Whole-band implication

To see 500 MHz contiguously with practical transition bands, the ADC/DDC/transport chain needs more than 500 MS/s of complex bandwidth in a single channel, or a bank of simultaneous sub-band channels whose *usable* passbands tile the allocation. At `sc16`, 500 MS/s alone is 2 GB/s per channel (before overhead). A sequential sweep has temporal blind spots: a short transmission can occur while the radio is tuned elsewhere. Report the sweep span, step size, dwell, retune/settling time and revisit period.

## 5. Firmware and software capture responsibilities

Here “firmware” should be used precisely: FPGA/device firmware controls clocking, DDC/decimation, channelization, packetization, timestamps, buffer behavior and sometimes onboard recording; host software (for example UHD plus GNU Radio or a UHD API application) configures the device and drains the stream. These layers interact, but a student MVP probably does not require custom FPGA firmware.

Capture software should:

- configure and log center frequency, analog bandwidth, rate, gain, channel, antenna port, clock/time source and IQ format;
- set a known clock source and use device timestamps where available;
- write in bounded buffers and monitor sustained disk throughput/CPU use;
- detect UHD overflows and sequence errors; mark a time gap or invalidate the affected interval;
- segment long captures, flush files safely and store metadata beside each segment;
- make AGC behavior explicit. Fixed gain simplifies comparisons; AGC can help with changing levels but changes the amplitude statistics and should be logged;
- distinguish calibrated power measurements from relative, uncalibrated digital power. For quantitative field strength, calibrate the whole antenna/cable/receiver chain and retain the calibration reference and gain state.

An overflow is not just a console warning: UHD documents that an internal receive buffer filling or a packet sequence error means data is missing. A timestamp discontinuity must therefore propagate into downstream labels/windows so later ML does not learn from stitched-together IQ as if it were continuous.

## 6. Suggested MVP scope

**Goal:** continuously monitor one known local N78 carrier, or scan a limited configured span and output timestamped IQ plus power/occupancy summaries.

**Baseline chain:** one receive antenna → optional fixed attenuation/preselector → one SDR receive channel (B210 class if available) → UHD/GNU Radio capture → short `sc16` IQ files with JSON metadata → offline PSD/waterfall and occupancy pipeline.

**Initial settings to validate, not assume:** center frequency based on a locally confirmed authorized carrier; 10–30.72 MS/s; one channel; modest fixed gain; short 5–30 s captures; a stable clock source; record-on-trigger or scheduled windows to limit data volume. Do not hard-code an arbitrary N78 center frequency without checking local spectrum authorization and the intended test signal.

**Minimum evidence to collect:**

- measured receiver noise floor and clipping rate at a few gain settings;
- known signal generator or legally available downlink capture to verify tuning and PSD scale;
- dropped-sample/overflow count and timestamp-gap handling under sustained capture;
- file write rate and CPU/memory load for each sample format/rate;
- sweep revisit interval if scanning multiple frequencies;
- repeatability across several captures with identical settings.

**Out of scope for Topic 1 MVP:** uninterrupted 500 MHz coverage, calibrated geolocation, decoding subscriber/user traffic, cell authentication, and custom FPGA development. Later signal classification needs its own evaluation and legitimate ground truth. A spectral anomaly by itself is not proof that a transmitter is rogue.

## 7. Research questions for the literature review

1. What is the difference between tuning range, instantaneous RF bandwidth, complex sample rate, occupied carrier bandwidth and usable flat bandwidth?
2. How do direct-conversion SDRs handle RF filtering, gain, I/Q imbalance, DC offset, LO leakage and ADC dynamic range?
3. What architecture best fits the monitoring objective: fixed-channel capture, swept spectrum, parallel channel bank, or wideband high-rate recording?
4. How do FPGA DDC/decimation and wire formats trade host throughput against bandwidth, resolution, and quantization noise?
5. How should timestamping, clock reference, retuning delay, and dropped samples be represented in capture metadata?
6. What calibration is required to compare power across frequency, gain, SDR units, antennas and time?
7. What sampling and capture bandwidth is needed for the particular NR carrier(s) of interest, including spectral margins and SCS?

## 8. Source notes

Primary standards and vendor documentation are preferred here; product specifications are capability ceilings and are not substitutes for measured system performance.

1. **3GPP TS 38.104 / ETSI TS 138 104, Release 19.3.0** — n78 band and base-station radio specifications; current version metadata and n78 range are available from the ETSI publication. [PDF](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/19.03.00_60/ts_138104v190300p.pdf), [3GPP specification record](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3202).
2. **3GPP TS 38.211 / ETSI TS 138 211, Release 19.1.0** — NR physical channels, modulation and numerology; defines Δf = 2^μ × 15 kHz. [PDF](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/19.01.00_60/ts_138211v190100p.pdf).
3. **Ettus USRP B210 product documentation** — 70 MHz–6 GHz tuning coverage and up to 56 MHz instantaneous bandwidth / 61.44 MS/s host stream. [Product page](https://www.ettus.com/all-products/ub210-kit/).
4. **Ettus bandwidth and sampling-rate guidance** — separates FPGA processing bandwidth from host interface throughput and describes supported rates. [Knowledge Base](https://kb.ettus.com/About_USRP_Bandwidths_and_Sampling_Rates).
5. **Analog Devices AD9361 data sheet/product page** — integrated 2×2 direct-conversion transceiver, 70 MHz–6 GHz RX range, 12-bit ADCs and channel bandwidth up to 56 MHz; product silicon specs should not be assumed as calibrated B210 performance. [Product documentation](https://www.analog.com/en/products/ad9361.html).
6. **Ettus USRP X310 documentation** — 200 MS/s conversion/processing and up to 160 MHz usable bandwidth per channel depending on daughterboard and interface. [Knowledge Base](https://kb.ettus.com/X300/X310).
7. **Ettus USRP N310 documentation** — four RX channels, up to 100 MHz per channel and configurable sample rates/clocking. [Knowledge Base](https://kb.ettus.com/N300/N310).
8. **UHD RX metadata documentation** — defines receive overflow/sequence error and the implication that samples are missing. [UHD API](https://uhd.readthedocs.io/en/uhd-4.10/structuhd_1_1rx__metadata__t.html).
9. **UHD stream arguments** — explains CPU/wire sample formats, including throughput versus quantization tradeoff for sc8. [UHD API](https://uhd.readthedocs.io/en/latest/structuhd_1_1stream__args__t.html).
10. **NI RF record-and-playback throughput note** — gives the common 4-byte-per-complex-sample estimate for 16-bit I/Q. [Engineering note](https://www.ni.com/en/solutions/aerospace-defense/electromagnetic-spectrum-operations/rf-record-playback-systems-enabling-real-world-signal-analysis-emulation.html).

## 9. Terms

- **N78:** 3GPP NR TDD band covering 3.3–3.8 GHz.
- **Instantaneous bandwidth:** bandwidth visible at one time without retuning.
- **Complex I/Q:** paired in-phase and quadrature samples representing a band around the receiver LO/center frequency.
- **DDC:** digital downconverter; shifts/filter/decimates sampled data in FPGA or software.
- **sc16 / sc8:** signed complex 16-bit or 8-bit I and Q data formats.
- **Overflow / sequence error:** receiver/transport data loss; the affected samples cannot be assumed continuous.
- **SCS / numerology:** NR subcarrier spacing; TS 38.211 defines it as powers of two times 15 kHz.
