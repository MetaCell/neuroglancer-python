//IMPORTS:
import "expect-puppeteer";
const { toMatchImageSnapshot } = require('jest-image-snapshot');
expect.extend({ toMatchImageSnapshot });
const path = require("path");
var scriptName = path.basename(__filename, ".js");
import * as selectors from "./selectors.js";
import { URL_START } from "./constants.js";

//PAGE INFO:
const baseURL = process.env.url || `${URL_START}#!%7B%22dimensions%22:%7B%22x%22:%5B7.48e-7%2C%22m%22%5D%2C%22y%22:%5B7.48e-7%2C%22m%22%5D%2C%22z%22:%5B0.000001%2C%22m%22%5D%2C%22t%22:%5B0.001%2C%22s%22%5D%7D%2C%22position%22:%5B29725.693359375%2C17808.77734375%2C11941.41796875%2C0%5D%2C%22crossSectionScale%22:36.59823444367803%2C%22projectionOrientation%22:%5B-0.13906417787075043%2C0.09761909395456314%2C0.22333434224128723%2C0.959819495677948%5D%2C%22projectionScale%22:17983.459691529064%2C%22projectionDepth%22:-0.7497788410238861%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22zarr://s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/%22%2C%22localDimensions%22:%7B%22c%27%22:%5B1%2C%22%22%5D%7D%2C%22localPosition%22:%5B0%5D%2C%22tab%22:%22source%22%2C%22shader%22:%22#uicontrol%20invlerp%20normalized%28range=%5B0%2C200%5D%29%5Cn#uicontrol%20transferFunction%20colormap%28range=%5B0%2C200%5D%29%5Cnvoid%20main%28%29%20%7B%5Cn%20%20emitRGBA%28colormap%28%29%29%3B%5Cn%7D%5Cn%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B67%2C201%5D%7D%2C%22colormap%22:%7B%22color%22:%22#1100ff%22%2C%22controlPoints%22:%5B%7B%22position%22:174%2C%22color%22:%7B%220%22:0%2C%221%22:0%2C%222%22:0%2C%223%22:0%7D%7D%2C%7B%22position%22:450%2C%22color%22:%7B%220%22:255%2C%221%22:255%2C%222%22:255%2C%223%22:255%7D%7D%5D%7D%7D%2C%22crossSectionRenderScale%22:0.47527330239775784%2C%22volumeRenderingDepthSamples%22:844.412726025728%2C%22name%22:%22fused.zarr%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22fused.zarr%22%7D%2C%22layout%22:%224panel%22%7D`;
const PAGE_WAIT = 3000;
const TIMEOUT = 60000;


//SNAPSHOT:
const SNAPSHOT_OPTIONS = {
  customSnapshotsDir: `./snapshot-tests/snapshots/${scriptName}`,
  comparisonMethod: "ssim",
  failureThresholdType: "percent",
  failureThreshold: 0.10,
};


//TESTS:

jest.setTimeout(300000);



describe("Test Suite for AIND Fusion Dataset", () => {
  beforeAll(async () => {
    

    await page.goto(baseURL);
    await page.waitForTimeout(3000);
    await page.waitForSelector(selectors.SIDE_PANEL);

  });

 
  describe("2D Canvas", () => {

    it("should navigate to rendering tab", async () => {
      console.log('Navigating to Rendering tab ...')
      await page.waitForSelector(selectors.SIDE_PANEL_TAB)
      await page.evaluate(() => {
        [...document.querySelectorAll('.neuroglancer-tab-label')].find(element => element.innerText === 'Rendering').click();
      });
      await page.waitForTimeout(1000);
      await page.waitForSelector(selectors.RENDERING_TAB_CONTROLS)
      const rendering_options = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options.length).toBe(6);
      console.log('Tab reached')
    })

    it("should wait for resolution slices to be displayed", async () => {
      console.log('Waiting for resolution slices to load ...')
      try {
        const isValueComplete = async () => {
          const { valueBeforeSlash, valueAfterSlash } = await page.$eval('div[title="Number of chunks rendered"]', (element) => {
            const textContent = element.textContent.trim();
            const [valueBeforeSlash, valueAfterSlash] = textContent.split('/').map(part => part.trim());
            
            return { valueBeforeSlash, valueAfterSlash };
          });
          return valueBeforeSlash >= (1 / 2) * valueAfterSlash;
        };

        const maxRetries = 50;
        let retries = 0;
    
        while (retries < maxRetries) {
          if (await isValueComplete()) {
            console.log('Value is reached. Continuing with the next steps.');
            break;
          }
          await page.waitForTimeout(4000); 
          retries++;
        }
    
        if (retries === maxRetries) {
          throw new Error('Timeout: Value did not become the expected within the specified time.');
        }
      } catch (error) {
        console.error('An error occurred:', error.message);
        throw error;
      }

    })


    it("should take screenshot of main canvas", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Main_canvas',
      });
      await page.waitForTimeout(1000 * 3);
    });
  })

  describe("Canvas with Volume Rendering", () => {

    it("should enable volume rendering", async () => {
      console.log('Enabling Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('on');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES)
      const rendering_options_afterVolume = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options_afterVolume.length).toBe(7);
      console.log('Volume Rendering enabled')
    });

    it("should wait for 3D resolution samples to be displayed", async () => {
      console.log('Waiting for 3D resolution samples to load ...')
      await page.waitForSelector(selectors.RESOLUTION_SAMPLES)
      try {
        const isValueComplete = async () => {
          const { valueBeforeSlash, valueAfterSlash } = await page.$eval('.neuroglancer-tab-content.neuroglancer-image-dropdown > div > .neuroglancer-layer-control-container.neuroglancer-layer-options-control-container > .neuroglancer-render-scale-widget.neuroglancer-layer-control-control > .neuroglancer-render-scale-widget-legend > div[title="Number of chunks rendered"]', (element) => {
            const textContent = element.textContent.trim();
            const [valueBeforeSlash, valueAfterSlash] = textContent.split('/').map(part => part.trim());
            
            return { valueBeforeSlash, valueAfterSlash };
          });
          
          return valueBeforeSlash >= (1 / 4.5) * valueAfterSlash;
        };

        const maxRetries = 50;
        let retries = 0;
    
        while (retries < maxRetries) {
          if (await isValueComplete()) {
            console.log('Value is reached. Continuing with the next steps.');
            break;
          }
          await page.waitForTimeout(4000); 
          retries++;
        }
    
        if (retries === maxRetries) {
          throw new Error('Timeout: Value did not become the expected within the specified time.');
        }
      } catch (error) {
        console.error('An error occurred:', error.message);
        throw error;
      }
    })

    
    it("should take screenshot of main canvas with 3D", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Main_canvas_w_3D',
      });
      await page.waitForTimeout(1000 * 3);
    });

  })

  
  describe("Canvas with Max Volume Rendering", () => {

    it("should enable max volume rendering", async () => {
      console.log('Enabling Max Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('max');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES)
      const rendering_options_afterVolume = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options_afterVolume.length).toBe(7);
      console.log('Max Volume Rendering enabled')
    })

    it("should take screenshot of main canvas with Max 3D Rendering", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Max_3D_Rendering',
      });
      await page.waitForTimeout(1000 * 3);
    });

  });

  describe("Canvas with colored 2D + 3D", () => {

    it("should change the color map of the 3D rendering", async () => {
      console.log('Changing color map ...')
      await page.waitForSelector(selectors.COLORMAP)
      await page.waitForSelector(selectors.COLORMAP_COLOR_WIDGET)
      await page.$eval('#neuroglancer-tf-color-widget', (colorWidget) => {
        colorWidget.value = '#00ff00'; 
        const event = new Event('change', { bubbles: true });
        colorWidget.dispatchEvent(event);
      });
      await page.waitForTimeout(1000 * 3);

      await page.click(selectors.COLORMAP_KNEES)
      await page.waitForTimeout(1000 * 5);
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('on');

      console.log('Color map changed')
    })

    it("should take screenshot of main colored canvas with 3D", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'colored_canvas',
      });
      await page.waitForTimeout(1000 * 3);
    });

  })

});
