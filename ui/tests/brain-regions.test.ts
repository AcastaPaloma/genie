import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  regionMask,
  REGION_NAMES,
  isRegionName,
  type RegionManifest,
} from "../src/brain-regions.ts";

const PATH = "public/data/full-brain-783/regions.json";
const manifest: RegionManifest | null = fs.existsSync(PATH)
  ? JSON.parse(fs.readFileSync(PATH, "utf8"))
  : null;
const measured = manifest ? test : test.skip;

test("region names are recognised exactly", () => {
  for (const name of REGION_NAMES) assert.ok(isRegionName(name));
  assert.ok(!isRegionName("optic-lobe"));
  assert.ok(!isRegionName(null));
  assert.ok(!isRegionName(""));
});

measured("central and optic partition every released neuron", () => {
  const m = manifest!;
  assert.equal(m.regions.all.neurons, m.neuronCount);
  assert.equal(
    m.regions.central.neurons + m.regions.optic.neurons,
    m.neuronCount,
  );
  assert.equal(
    m.regions.central.vertices + m.regions.optic.vertices,
    m.regions.all.vertices,
  );
  assert.equal(
    m.regions.central.edges + m.regions.optic.edges,
    m.regions.all.edges,
  );
});

measured(
  "the rebuilt mask selects exactly the neurons the manifest counts",
  () => {
    const m = manifest!;
    const central = regionMask(m, "central")!,
      optic = regionMask(m, "optic")!;
    assert.equal(central.length, m.neuronCount);
    const drawn = (mask: Float32Array) =>
      mask.reduce((sum, value) => sum + value, 0);
    assert.equal(drawn(central), m.regions.central.neurons);
    assert.equal(drawn(optic), m.regions.optic.neurons);
    // Every neuron is in exactly one of the two, and each value is a clean 0 or 1.
    for (let i = 0; i < central.length; i++)
      assert.equal(central[i] + optic[i], 1);
    assert.equal(
      regionMask(m, "all"),
      null,
      "the complete brain needs no mask",
    );
  },
);

measured(
  "the central region excludes the optic lobes and keeps the rest",
  () => {
    const m = manifest!;
    assert.deepEqual(m.regions.central.excludedSuperClasses, ["optic"]);
    // Withholding the lobes must remove a large majority of optic-lobe cable while
    // keeping most central cable, or the view has not actually changed.
    assert.ok(
      m.regions.optic.vertices > 9e7,
      "the optic lobes should hold ~99M vertices",
    );
    assert.ok(
      m.regions.central.vertices > 1.6e8,
      "the central brain should keep ~170M vertices",
    );
    // The framing box must be materially tighter than the whole brain, otherwise
    // the camera would not zoom in on the region at all.
    const width = (r: "central" | "all") =>
      m.regions[r].framingBounds[1][0] - m.regions[r].framingBounds[0][0];
    assert.ok(
      width("central") < width("all") * 0.85,
      "central framing should be tighter than the full brain",
    );
  },
);

measured("a malformed run is rejected rather than silently clipped", () => {
  const broken = {
    ...manifest!,
    opticRuns: [[manifest!.neuronCount - 1, 5]] as [number, number][],
  };
  assert.throws(() => regionMask(broken, "central"), /outside the anatomy/);
});
