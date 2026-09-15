// Adapted from NeLy-EPFL/flygym, Apache-2.0. Ported CPG/recorded-step controller; not VNC simulation.
const TAU = Math.PI * 2;
export class Controller {
  constructor(meta) {
    this.dt = meta.timestep;
    this.legs = meta.control.leg_order;            // 6 leg names
    this.cmap = meta.ctrl_index_by_leg_dof;        // [6][7] -> ctrl index
    this.adh = meta.adhesion;                       // [6] -> adhesion ctrl index
    this.tripodMap = meta.control.tripod_map;       // [6] -> 0/1
    const cpg = meta.control.cpg;
    this.freqs0 = cpg.intrinsic_freqs.slice();      // base |freq| per leg
    this.W = cpg.coupling_weights;                  // [6][6]
    this.PB = cpg.phase_biases;                     // [6][6]
    this.conv = cpg.convergence_coefs;              // [6]
    this.phaseInc = (this.dt / meta.control.leg_step_time) * TAU;

    const pp = meta.preprogrammed;
    this.N = pp.n_samples;
    this.tab = this.legs.map((l) => pp.legs[l]);    // {angles:[N][7], neutral:[7], swing:[2]}

    // scratch
    this._cpgAmps = new Float64Array(6);
    this._cpgFreqs = new Float64Array(6);
    this._d6 = new Float64Array(6);
    this._a7 = new Float64Array(7);
    this.reset();
  }

  reset() {
    this._seedIndex = 0;
    this.phases = new Float64Array(6).map(() => [0, Math.PI, 0, Math.PI, 0, Math.PI][this._seedIndex++ % 6]); // CPG phases
    this.mags = new Float64Array(6);                                  // CPG magnitudes
    this.legPhases = new Float64Array(6);                             // single-leg
    this.stepDir = new Float64Array(6);
    this.tripodPhases = new Float64Array(2);                          // tripod groups
    this.tripodDir = new Float64Array(2);
  }

  // Joint angles for one leg at a phase / magnitude, by periodic-lerp of the
  // baked table: angle = neutral + magnitude * (table(phase) - neutral).
  _anglesInto(li, phase, mag, out) {
    const t = this.tab[li], N = this.N;
    let x = ((phase % TAU) + TAU) % TAU / TAU * N;
    const i0 = Math.floor(x) % N, i1 = (i0 + 1) % N, f = x - Math.floor(x);
    const a0 = t.angles[i0], a1 = t.angles[i1], nu = t.neutral;
    for (let d = 0; d < 7; d++) {
      const samp = a0[d] * (1 - f) + a1[d] * f;
      out[d] = nu[d] + mag * (samp - nu[d]);
    }
  }

  _adhesionOn(li, phase) {
    const [s, e] = this.tab[li].swing;             // swing = adhesion OFF
    const p = ((phase % TAU) + TAU) % TAU;
    return !(p > s && p < e);
  }

  _writeLeg(ctrl, li, phase, mag) {
    this._anglesInto(li, phase, mag, this._a7);
    const row = this.cmap[li];
    for (let d = 0; d < 7; d++) ctrl[row[d]] = this._a7[d];
    ctrl[this.adh[li]] = this._adhesionOn(li, phase) ? 1 : 0;
  }

  // Level 1: descending signal action=[gainL,gainR] modulates CPG amplitude
  // (|action|) and stepping direction (sign), then one Euler integration step.
  stepCPG(ctrl, gainL, gainR) {
    const amps = this._cpgAmps, freqs = this._cpgFreqs;
    const aL = Math.abs(gainL), aR = Math.abs(gainR);
    amps[0] = amps[1] = amps[2] = aL; amps[3] = amps[4] = amps[5] = aR;
    const sL = gainL > 0 ? 1 : -1, sR = gainR > 0 ? 1 : -1;
    for (let i = 0; i < 6; i++) freqs[i] = this.freqs0[i] * (i < 3 ? sL : sR);

    // dtheta = 2pi*freq + sum_j mags_j*W_ij*sin(theta_j - theta_i - PB_ij);  dr = conv*(amp - r)
    const ph = this.phases, mg = this.mags, dt = this.dt;
    const dph = this._d6;
    for (let i = 0; i < 6; i++) {
      let coupling = 0;
      for (let j = 0; j < 6; j++)
        coupling += mg[j] * this.W[i][j] * Math.sin(ph[j] - ph[i] - this.PB[i][j]);
      dph[i] = TAU * freqs[i] + coupling;
    }
    for (let i = 0; i < 6; i++) {
      ph[i] += dph[i] * dt;
      mg[i] += this.conv[i] * (amps[i] - mg[i]) * dt;
    }
    for (let i = 0; i < 6; i++) this._writeLeg(ctrl, i, ph[i], mg[i]);
  }

  // Shared per-step state machine for the on-demand modes: a leg/tripod at rest
  // (phase<=0) starts a step when its action is non-zero, runs the cycle to
  // completion (forward to 2pi, or backward to 0), then returns to rest.
  _advance(phaseArr, dirArr, i, act) {
    if (phaseArr[i] >= TAU || (phaseArr[i] <= 0 && dirArr[i] < 0)) {
      phaseArr[i] = 0; dirArr[i] = 0;
    } else if (phaseArr[i] <= 0) {
      if (act > 0) { phaseArr[i] += this.phaseInc; dirArr[i] = 1; }
      else if (act < 0) { phaseArr[i] = TAU - this.phaseInc; dirArr[i] = -1; }
    } else {
      phaseArr[i] += this.phaseInc * dirArr[i];
    }
  }

  // Level 3: action is a 6-vector (one trigger per leg).
  stepSingle(ctrl, action) {
    for (let i = 0; i < 6; i++) {
      this._advance(this.legPhases, this.stepDir, i, action[i]);
      this._writeLeg(ctrl, i, this.legPhases[i], 1);
    }
  }

  // Level 2: action is a 2-vector (one trigger per tripod group).
  stepTripod(ctrl, action) {
    for (let g = 0; g < 2; g++) this._advance(this.tripodPhases, this.tripodDir, g, action[g]);
    for (let i = 0; i < 6; i++)
      this._writeLeg(ctrl, i, this.tripodPhases[this.tripodMap[i]], 1);
  }
}

