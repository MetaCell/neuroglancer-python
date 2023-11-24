//IMPORTS:
// import "expect-puppeteer";
const { toMatchImageSnapshot } = require('jest-image-snapshot');
expect.extend({ toMatchImageSnapshot });
const puppeteer = require('puppeteer')
const path = require("path");
var scriptName = path.basename(__filename, ".js");
// import * as selectors from "./selectors";

//PAGE INFO:
const baseURL = process.env.url || "https://neuroglancer.dev.metacell.us/#!%7B%22dimensions%22:%7B%22x%22:%5B7.48e-7%2C%22m%22%5D%2C%22y%22:%5B7.48e-7%2C%22m%22%5D%2C%22z%22:%5B0.000001%2C%22m%22%5D%2C%22t%22:%5B0.001%2C%22s%22%5D%7D%2C%22position%22:%5B29725.693359375%2C17808.77734375%2C11941.41796875%2C0%5D%2C%22crossSectionScale%22:36.59823444367803%2C%22projectionOrientation%22:%5B-0.13906417787075043%2C0.09761909395456314%2C0.22333434224128723%2C0.959819495677948%5D%2C%22projectionScale%22:17983.459691529064%2C%22projectionDepth%22:-0.7497788410238861%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22zarr://s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/%22%2C%22localDimensions%22:%7B%22c%27%22:%5B1%2C%22%22%5D%7D%2C%22localPosition%22:%5B0%5D%2C%22tab%22:%22source%22%2C%22shader%22:%22#uicontrol%20invlerp%20normalized%28range=%5B0%2C200%5D%29%5Cn#uicontrol%20transferFunction%20colormap%28range=%5B0%2C200%5D%29%5Cnvoid%20main%28%29%20%7B%5Cn%20%20emitRGBA%28colormap%28%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B67%2C201%5D%7D%2C%22colormap%22:%7B%22color%22:%22#1100ff%22%2C%22controlPoints%22:%5B%7B%22position%22:174%2C%22color%22:%7B%220%22:0%2C%221%22:0%2C%222%22:0%2C%223%22:0%7D%7D%2C%7B%22position%22:450%2C%22color%22:%7B%220%22:255%2C%221%22:255%2C%222%22:255%2C%223%22:255%7D%7D%5D%7D%7D%2C%22crossSectionRenderScale%22:0.47527330239775784%2C%22volumeRenderingDepthSamples%22:844.412726025728%2C%22name%22:%22fused.zarr%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22fused.zarr%22%7D%2C%22layout%22:%224panel%22%7D";
const PAGE_WAIT = 3000;
const TIMEOUT = 60000;


const SHADER_FUNCTIOON = `#uicontrol float gain slider(min=0, max=10, default=1.0)
#uicontrol invlerp normalized(range=[0,255], clamp=true)
#uicontrol vec3 color color(default="white")
void main() {
    float val = normalized();
    emitRGBA(vec4(color, val * gain));
    }"
`
//SNAPSHOT:
const SNAPSHOT_OPTIONS = {
  customSnapshotsDir: `./snapshot-tests/snapshots/${scriptName}`,
  comparisonMethod: "ssim",
  failureThresholdType: "percent",
  failureThreshold: 0.25,
};



//TESTS:

jest.setTimeout(300000);

let page;
let browser;

describe("Test Suite for Dataset", () => {
  beforeAll(async () => {
    browser = await puppeteer.launch({
      args: [
        "--no-sandbox",
        // `--window-size=1600,1000`,
        "--ignore-certificate-errors",
      ],
      headless: false,
      devtools: false,
      defaultViewport: {
        width: 1600,
        height: 1000,
      },
    });

    page = await browser.newPage();
    await page.goto(baseURL);
    await page.waitForTimeout(3000);
    await page.waitForSelector(".neuroglancer-side-panel");

    // page.on("response", (response) => {
    //   const client_server_errors = range(90, 400);
    //   for (let i = 0; i < client_server_errors.length; i++) {
    //     expect(response.status()).not.toBe(client_server_errors[i]);
    //   }
    // });
  });

  afterAll(() => {
    browser.close();
  });
  describe("Add Data", () => {

    it("should take screenshot of main canvas", async () => {
      const canvas = await page.waitForSelector('.neuroglancer-layer-group-viewer', {hidden:false});
      await page.waitForTimeout(1000 * 6);
      // await page.waitForNavigation({ waitUntil: "networkidle2", timeout: 60000 });
      const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Main canvas',
      });
      await page.waitForTimeout(1000 * 3);
    });

    it("should navigate to rendering tab", async () => {
      await page.evaluate(() => {
        [...document.querySelectorAll('.neuroglancer-tab-label')].find(element => element.innerText === 'Rendering').click();
      });
      await page.waitForTimeout(1000);
      await page.waitForSelector('.neuroglancer-layer-control-control')
      const rendering_options = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options.length).toBe(6);
    })

    it("should enable volume rendering", async () => {

      await page.waitForSelector('select.neuroglancer-layer-control-control')
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector('option[value="off"]')
      await page.waitForSelector('option[value="on"]')
      await page.waitForSelector('option[value="max"]')
      await dropdown_buttons[1].select('on');
      await page.waitForTimeout(2000);
      await page.waitForSelector('div[title="Target resolution of data in screen pixels"]')
      const rendering_options_afterVolume = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options_afterVolume.length).toBe(7);

    });
    


  });



});
