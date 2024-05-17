//IMPORTS:
import "expect-puppeteer";
const { toMatchImageSnapshot } = require('jest-image-snapshot');
expect.extend({ toMatchImageSnapshot });
const path = require("path");
var scriptName = path.basename(__filename, ".js");
import * as selectors from "./selectors.js";
import { URL_START, THRESHOLD, TIMEOUT } from "./constants.js";

//PAGE INFO:
const baseURL = process.env.url || `${URL_START}#!%7B"dimensions":%7B"x":%5B7.48e-7%2C"m"%5D%2C"y":%5B7.48e-7%2C"m"%5D%2C"z":%5B0.000001%2C"m"%5D%2C"t":%5B0.001%2C"s"%5D%7D%2C"position":%5B29725.693359375%2C17808.77734375%2C11941.41796875%2C0%5D%2C"crossSectionScale":36.59823444367803%2C"projectionOrientation":%5B-0.13906417787075043%2C0.09761909395456314%2C0.22333434224128723%2C0.959819495677948%5D%2C"projectionScale":17983.459691529064%2C"projectionDepth":-0.7497788410238861%2C"layers":%5B%7B"type":"image"%2C"source":"zarr://s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/"%2C"localDimensions":%7B"c%27":%5B1%2C""%5D%7D%2C"localPosition":%5B0%5D%2C"tab":"source"%2C"shader":"#uicontrol%20invlerp%20normalized%28range=%5B0%2C200%5D%29%5Cnvoid%20main%28%29%20%7B%5Cn%20%20emitGrayscale%28normalized%28%29%29%3B%5Cn%7D%5Cn"%2C"shaderControls":%7B"normalized":%7B"range":%5B67%2C201%5D%7D%7D%2C"crossSectionRenderScale":0.47527330239775784%2C"volumeRenderingDepthSamples":844.412726025728%2C"name":"fused.zarr"%7D%5D%2C"showSlices":false%2C"selectedLayer":%7B"visible":true%2C"layer":"fused.zarr"%7D%2C"layout":"4panel"%7D`;


//SNAPSHOT:
const SNAPSHOT_OPTIONS = {
  customSnapshotsDir: `./snapshot-tests/snapshots/${scriptName}`,
  comparisonMethod: "ssim",
  failureThresholdType: "percent",
  failureThreshold: THRESHOLD,
};


//TESTS:

jest.setTimeout(TIMEOUT);



describe("Test Suite for AIND Fusion Dataset", () => {
  beforeAll(async () => {
    await page.goto(baseURL);
    await page.waitForTimeout(3000);
    await page.waitForSelector(selectors.SIDE_PANEL, { hidden: false });

  });


  describe("2D Canvas", () => {

    it("should navigate to rendering tab", async () => {
      console.log('Navigating to Rendering tab ...')
      await page.waitForSelector(selectors.SIDE_PANEL_TAB, { hidden: false })
      await page.evaluate(() => {
        [...document.querySelectorAll('.neuroglancer-tab-label')].find(element => element.innerText === 'Rendering').click();
      });
      await page.waitForTimeout(1000);
      await page.waitForSelector(selectors.RENDERING_TAB_CONTROLS, { hidden: false })
      const rendering_options = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options.length).toBe(5);
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
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, { hidden: false });
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
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS, { hidden: false })
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
      await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE, { hidden: false })
      await page.waitForSelector(selectors.ON_VALUE, { hidden: false })
      await page.waitForSelector(selectors.MAX_VALUE, { hidden: false })
      await dropdown_buttons[1].select('on');
      await page.waitForFunction((selector) => {
        const dropdowns = Array.from(document.querySelectorAll(selector));
        return dropdowns[1] && dropdowns[1].value === 'on';
      }, {}, '.neuroglancer-layer-control-container.neuroglancer-layer-options-control-container > select.neuroglancer-layer-control-control');
      await page.waitForSelector(selectors.RESOLUTION_SLICES, { hidden: false })
      await page.waitForSelector('.neuroglancer-layer-control-container.neuroglancer-layer-options-control-container')

      console.log('Volume Rendering enabled')
    });

    it("should wait for 3D resolution samples to be displayed", async () => {
      console.log('Waiting for 3D resolution samples to load ...')
      await page.waitForSelector(selectors.RESOLUTION_SAMPLES, { hidden: false })
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


    it("should take screenshot of main canvas with 3D", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, { hidden: false });
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
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS, { hidden: false })
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
      await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE, { hidden: false })
      await page.waitForSelector(selectors.ON_VALUE, { hidden: false })
      await page.waitForSelector(selectors.MAX_VALUE, { hidden: false })
      await dropdown_buttons[1].select('max');
      await page.waitForFunction((selector) => {
        const dropdowns = Array.from(document.querySelectorAll(selector));
        return dropdowns[1] && dropdowns[1].value === 'max';
      }, {}, '.neuroglancer-layer-control-container.neuroglancer-layer-options-control-container > select.neuroglancer-layer-control-control');
      await page.waitForSelector(selectors.RESOLUTION_SAMPLES, { hidden: false })

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

      console.log('Max Volume Rendering enabled')
    })

    it("should maximize 3D panel", async () => {
      await page.waitForSelector('button[title="Switch to 3d layout."]', { hidden: false });
      await page.click('button[title="Switch to 3d layout."]');
      await page.waitForSelector('button[title="Switch to 4panel layout."]', { hidden: false });
    })

    it("should take screenshot of main canvas with Max 3D Rendering", async () => {
      await page.waitForSelector(selectors.IMAGE_CANVAS, { hidden: false });

      // await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Max_3D_Rendering',
      });
      await page.waitForTimeout(1000 * 3);

    });

    it("should reset to 4 panel layout", async () => {
      await page.waitForSelector('button[title="Switch to 4panel layout."]', { hidden: false });
      await page.click('button[title="Switch to 4panel layout."]');
      await page.waitForSelector('button[title="Switch to 3d layout."]', { hidden: false });
    })

  });

  describe("Canvas with Min Volume Rendering", () => {

    it("should enable min volume rendering", async () => {
      console.log('Enabling Min Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
      await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE, { hidden: false })
      await page.waitForSelector(selectors.ON_VALUE, { hidden: false })
      await page.waitForSelector(selectors.MAX_VALUE, { hidden: false })
      await page.waitForSelector(selectors.MIN_VALUE, { hidden: false })
      await dropdown_buttons[1].select('min');
      await dropdown_buttons[1].select('min');
      await page.waitForFunction((selector) => {
        const dropdowns = Array.from(document.querySelectorAll(selector));
        return dropdowns[1] && dropdowns[1].value === 'min';
      }, {}, '.neuroglancer-layer-control-container.neuroglancer-layer-options-control-container > select.neuroglancer-layer-control-control');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES, { hidden: false })

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


      console.log('Min Volume Rendering enabled')
    })

    it("should maximize 3D panel", async () => {
      await page.waitForSelector('button[title="Switch to 3d layout."]', { hidden: false });
      await page.click('button[title="Switch to 3d layout."]');
      await page.waitForSelector('button[title="Switch to 4panel layout."]', { hidden: false });
    })

    it("should take screenshot of main canvas with Min 3D Rendering", async () => {
      await page.waitForSelector(selectors.IMAGE_CANVAS, { hidden: false });

      // await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Min_3D_Rendering',
      });
      await page.waitForTimeout(1000 * 3);

    });

    it("should reset to 4 panel layout", async () => {
      await page.waitForSelector('button[title="Switch to 4panel layout."]', { hidden: false });
      await page.click('button[title="Switch to 4panel layout."]');
      await page.waitForSelector('button[title="Switch to 3d layout."]', { hidden: false });
    })
  })

  describe.skip("Canvas with colored 2D + 3D", () => {

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
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, { hidden: false });
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
