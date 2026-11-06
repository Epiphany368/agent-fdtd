import {test,expect} from '@playwright/test';

test('angular-spectrum panel propagates a stored plane and reports focus, spectrum and aliasing',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();
 await expect(page.getByLabel('dimension',{exact:true})).toHaveValue('3d');
 await page.locator('[data-add="field"]').click();
 await page.locator('[data-path="center.0"]').fill('1.2');await page.locator('[data-path="center.0"]').press('Tab');
 await page.locator('[data-action="region"]').first().click();
 const gpu=page.getByLabel('GPU',{exact:true});
 if(await gpu.isEnabled())await gpu.uncheck();
 else{ // no CUDA device: the switch is disabled and the resource stays 'GPU if available'; pin the CPU explicitly
  const toggle=page.locator('[data-advanced-toggle]');
  if(await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();
  await page.getByLabel('resource',{exact:true}).selectOption('cpu');
 }
 await expect(page.getByLabel('resource',{exact:true})).toHaveValue('cpu');
 await page.getByLabel('time steps',{exact:true}).fill('200');await page.getByLabel('time steps',{exact:true}).press('Tab');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const job=await (await submitted).json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await page.locator('#results-tree [data-action="propagation"]').click();
 const dialog=page.locator('.propagation-dialog');
 await expect(dialog).toBeVisible();
 await expect(dialog.locator('h2')).toHaveText('Angular spectrum');
 await expect(dialog.locator('[data-plane-note]')).toContainText('+x inferred from the sources');
 await expect(dialog.getByLabel('Propagation kind')).toHaveValue('section:y');
 expect(await dialog.locator('[data-key="kind"] option').allTextContents()).toEqual(['Section yx','Section zx','Plane parallel to the monitor']);
 await dialog.getByLabel('Distance stop (µm)').fill('4');
 await dialog.getByLabel('Planes').fill('41');
 const calculated=page.waitForResponse(r=>r.url().endsWith('/propagate')&&r.request().method()==='POST');
 await dialog.locator('[data-calculate]').click();
 const response=await calculated;
 expect(response.status()).toBe(200);
 const body=response.request().postDataJSON();
 expect(body).toMatchObject({monitor:expect.any(String),kind:'section',axis:'y',z_start_um:0,z_stop_um:4,planes:41,index:1,pad:2,direction:'auto'});
 const result=await response.json();
 expect(result.device).toBe('cpu');
 expect(result.geometry.axes).toEqual(['x','y']);
 expect(result.image.length).toBe(41);
 await expect(dialog.locator('[role="status"]')).toContainText('angular spectrum section');
 await expect(dialog.locator('.focus-report')).toContainText(/Peak intensity .* at .* µm from the plane \(x = .* µm\), y = .* µm; FWHM along y/);
 await expect(dialog.locator('.spectrum-report')).toContainText('Largest representable angle 90.00°');
 await expect(dialog.locator('.spectrum-report')).toContainText('Direction +normal (inferred from the sources), CPU, Ex, Ey, Ez');
 await expect(dialog.locator('.aliasing-warning')).toHaveCount(0);
 await dialog.getByLabel('Propagation kind').selectOption('plane');
 await dialog.getByLabel('Plane distance (µm)').fill('2');
 await dialog.locator('[data-calculate]').click();
 await expect(dialog.locator('[role="status"]')).toContainText('angular spectrum volume');
 await expect(dialog.locator('.focus-report')).toContainText('on the plane 2.000 µm away');
 // A dense exterior makes the 0.1 um monitor spacing exceed half the wavelength: the panel warns about aliasing.
 await dialog.getByLabel('Exterior index').fill('16');
 await dialog.locator('[data-calculate]').click();
 await expect(dialog.locator('.aliasing-warning')).toContainText('exceeds half the wavelength in the exterior');
 await expect(dialog.locator('.spectrum-report')).not.toContainText('Largest representable angle 90.00°');
 await page.screenshot({path:'results/ui-angular-spectrum.png',fullPage:true});
 // The flux dialog reaches the same panel.
 await dialog.locator('[data-close]').click();
 await expect(page.locator('.propagation-dialog')).toHaveCount(0);
 await page.locator('[data-action="flux-results"]').click();
 await page.locator('[data-propagate]').click();
 await expect(page.locator('.propagation-dialog')).toBeVisible();
 expect(errors).toEqual([]);
});
