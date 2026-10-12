import React, { useState } from "react";
import { DataComponent, EvidenceChart, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./evaluation.css";

const detectors = ["safe", "universal_fake_detect", "ssp", "bfree", "aeroblade"];
const rawLabels = {
  safe: "Synthetic probability",
  universal_fake_detect: "Synthetic probability",
  ssp: "Human probability",
  bfree: "Synthetic logit",
  aeroblade: "Minimum reconstruction error",
};
const number = value => Number(value).toPrecision(5);
const interval = value => value.map(v => Number(v).toFixed(3)).join(" – ");

export function ReportContent() {
  const { reviewedRows } = useDataApp();
  const recipe = reviewedRows("recipe")[0];
  const [codec, setCodec] = useState("png");
  const [group, setGroup] = useState("all");
  const [comparison, setComparison] = useState("random");
  const [regime, setRegime] = useState("png_locked");
  const regimeLabel = regime === "png_locked" ? "PNG thresholds" : "codec recalibration";
  const [detail, setDetail] = useState(null);
  const matrix = reviewedRows("matrix").filter(r => r.codec === codec && (group === "all" || r.group === group));
  const metricRows = reviewedRows("metrics");
  const metrics = metricRows.filter(r => r.codec === codec && r.group === comparison && r.regime === regime);
  const roc = metrics.flatMap(m => m.roc.fpr.map((fpr, i) => ({detector:m.detector, fpr, tpr:m.roc.tpr[i]})));
  const rates = metrics.map(m => ({detector:m.detector, detectionRate:m.detection_rate, falsePositiveRate:m.false_positive_rate}));
  const primaryPNG = metricRows.filter(r => r.codec === "png" && r.regime === "png_locked");
  const bfreeRandom = primaryPNG.find(r => r.detector === "bfree" && r.group === "random");
  const bfreePrompted = primaryPNG.find(r => r.detector === "bfree" && r.group === "prompted");
  return <article className="report-page detector-report">
    <RichNarrative id="evaluation-method" value={`# Image detector evaluation\n\n${recipe.test_count} real photographs, ${recipe.test_count} random embedding generations and ${recipe.test_count} prompted controls. Thresholds use ${recipe.calibration_count} separate real calibration photographs. All five detectors score PNG and JPEG95. Random outputs are retained without selection.\n\nMatrix colors are percentiles relative to each detector’s PNG calibration scores, **not probabilities**. Decisions use the frozen PNG thresholds for both codecs. Results are conditional on one-step SDXL-Turbo with guidance zero and independently reset diffusion seed 0.`} />
    <DataComponent id="evaluation-key-finding" queryId="metrics" title="Observed result in this evaluation" sourceRows={primaryPNG}>
      <p data-reviewed-rows>B-Free detects {(100*bfreeRandom.detection_rate).toFixed(0)}% of random and {(100*bfreePrompted.detection_rate).toFixed(0)}% of prompted PNG generations, with {(100*bfreeRandom.false_positive_rate).toFixed(0)}% false positives on the real test photographs. This comparison applies to the fixed SDXL-Turbo/noise-0 recipe and these image groups.</p>
    </DataComponent>
    <div className="evaluation-filters">
      <label>Codec <select aria-label="Codec" value={codec} onChange={e => {setCodec(e.target.value);setDetail(null);}}><option value="png">PNG</option><option value="jpeg95">JPEG95</option></select></label>
      <label>Image group <select aria-label="Image group" value={group} onChange={e => {setGroup(e.target.value);setDetail(null);}}>{["all","real","random","prompted"].map(g => <option key={g}>{g}</option>)}</select></label>
      <label>ROC comparison <select aria-label="ROC comparison" value={comparison} onChange={e => setComparison(e.target.value)}><option>random</option><option>prompted</option></select></label>
      <label>Metric thresholds <select aria-label="Metric thresholds" value={regime} onChange={e => setRegime(e.target.value)}><option value="png_locked">Frozen PNG thresholds (primary)</option><option value="codec_calibrated">Codec recalibration (secondary)</option></select></label>
    </div>
    <DataComponent id="image-detector-matrix" queryId="matrix" title={`${matrix.length} images × five detectors · ${codec}`} sourceRows={matrix} displayRows={matrix}>
      <div className="evaluation-scroll matrix-scroll" data-reviewed-rows>
        <table className="evaluation-matrix"><thead><tr><th scope="col">Image</th><th scope="col">Group</th>{detectors.map(d => <th key={d} scope="col">{d}</th>)}</tr></thead>
        <tbody>{matrix.map(row => <tr key={row.image_id}><th scope="row"><button className="image-button" onClick={() => setDetail(row)} aria-label={`Inspect ${row.image_id}`}><img src={row.thumbnail} alt="" loading="lazy" width="64" height="64" />{row.image_id}</button></th><td>{row.group}</td>{detectors.map(d => {
          const s = row.detectors[d];
          const color = Math.floor(s.calibration_percentile / 20);
          return <td key={d} className={`percentile-${color}`}><button onClick={() => setDetail(row)} aria-label={`Inspect ${row.image_id}, ${d}`} title={`${rawLabels[d]}: ${number(s.raw_score)}; synthetic score: ${number(s.synthetic_score)}`}><strong>{s.calibration_percentile.toFixed(1)} pct</strong><span>{s.predicted_fake ? "FAKE" : "REAL"}</span><small>raw {number(s.raw_score)}</small></button></td>;
        })}</tr>)}</tbody></table>
      </div>
      {detail && <section className="image-detail" data-reviewed-rows aria-label="Selected image details">
        <button onClick={() => setDetail(null)}>Close details</button>
        <h3>{detail.image_id} · {detail.group} · {detail.codec}</h3>
        <a href={detail.image_url} target="_blank" rel="noreferrer"><img src={detail.image_url} alt={detail.image_id} width="512" height="512" /></a>
        <p>{detail.detectors.safe.prompt || "No textual prompt for this image."}</p>
        <div className="evaluation-scroll"><table><thead><tr><th>Detector</th><th>Native score (meaning)</th><th>Synthetic score</th><th>PNG calibration percentile</th><th>Decision</th></tr></thead><tbody>{detectors.map(d => {const s=detail.detectors[d];return <tr key={d}><th>{d}</th><td>{number(s.raw_score)} · {rawLabels[d]}</td><td>{number(s.synthetic_score)}</td><td>{s.calibration_percentile.toFixed(1)}</td><td>{s.predicted_fake ? "fake" : "real"}</td></tr>;})}</tbody></table></div>
        {detail.detectors.ssp.patch_path && <><h4>SSP selected 32 × 32 patch</h4>
        <img className="ssp-patch" src={`../${detail.detectors.ssp.patch_path}`} alt="SSP selected simplest patch" width="128" height="128" />
        <p>Candidate {detail.detectors.ssp.selected_index + 1} of 64. Coordinates: {JSON.stringify(detail.detectors.ssp.coordinates[detail.detectors.ssp.selected_index])}.</p></>}
      </section>}
    </DataComponent>
    <EvidenceChart id="detector-roc" queryId="metrics" title={`Real versus ${comparison} · ${codec} · ROC`} rows={roc} sourceRows={metrics} spec={{type:"line",x:"fpr",y:"tpr",series:"detector",stackable:false,xLabel:"False-positive rate",yLabel:"Detection rate"}} height={360} />
    <EvidenceChart id="detector-rates" queryId="metrics" title={`Detection and false-positive rates · ${codec} · ${regimeLabel}`} rows={rates} sourceRows={metrics} spec={{type:"bar",x:"detector",y:"detectionRate",fields:["detectionRate","falsePositiveRate"],stackable:false}} height={320} />
    <DataComponent id="detector-intervals" queryId="metrics" title={`Measured performance and 95% intervals · ${regimeLabel}`} sourceRows={metrics} displayRows={metrics}>
      <div className="evaluation-scroll" data-reviewed-rows><table><thead><tr><th>Detector</th><th>AUROC (bootstrap CI)</th><th>AP (bootstrap CI)</th><th>Detection (Wilson CI)</th><th>FPR (Wilson CI)</th><th>Balanced accuracy</th></tr></thead><tbody>{metrics.map(m => <tr key={m.detector}><th>{m.detector}</th><td>{m.auroc.toFixed(3)} [{interval(m.auroc_ci95)}]</td><td>{m.ap.toFixed(3)} [{interval(m.ap_ci95)}]</td><td>{m.detection_rate.toFixed(3)} [{interval(m.detection_ci95)}]</td><td>{m.false_positive_rate.toFixed(3)} [{interval(m.false_positive_ci95)}]</td><td>{m.balanced_accuracy.toFixed(3)}</td></tr>)}</tbody></table></div>
      <p>{recipe.bootstrap_samples.toLocaleString()} seeded paired image bootstrap resamples. Threshold uncertainty is conditional on frozen calibration. The image matrix always uses primary PNG thresholds; metric tables use the selected threshold regime. Static confusion/rate figures show the primary regime.</p>
    </DataComponent>
    <DataComponent id="scientific-comparison-figures" queryId="metrics" title="Scientific comparison figures" sourceRows={metrics}>
      <img className="comparison-figure" src={`figures/confusion-${comparison}-${codec}.png`} alt={`Confusion matrices for ${comparison}, ${codec}`} />
      <img className="comparison-figure" src={`figures/rates-${comparison}-${codec}.png`} alt="Detection and false-positive rates with Wilson intervals" />
      {detectors.map(d => <details key={d}><summary>{d}: distributions and compression changes</summary><img className="comparison-figure" src={`figures/distribution-${d}.png`} alt={`${d} distributions`} /><img className="comparison-figure" src={`figures/codec-drift-${d}.png`} alt={`${d} JPEG minus PNG score changes`} /></details>)}
      {["real","random","prompted"].map(g => <details key={g}><summary>{g}: correlation and disagreement</summary><img className="comparison-figure" src={`figures/agreement-${g}-${codec}.png`} alt={`${g} correlation and disagreement matrices`} /></details>)}
      <p><a href="stage_timings.csv">Separate cold loading, preprocessing, inference and persistence timings (CSV)</a> · <a href="generation_timing_history.json">Original generation and recovery reload timings (JSON)</a> · <a href="recovery_disclosure.json">Compatibility and provenance disclosure</a></p>
      {reviewedRows("runtime").length > 0 && <img className="comparison-figure" src="figures/runtime-memory.png" alt="Validated warm runtime and GPU memory" />}
    </DataComponent>
  </article>;
}
