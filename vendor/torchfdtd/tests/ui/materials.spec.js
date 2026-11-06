import { test, expect } from '@playwright/test';
import fs from 'node:fs';

test('edit Drude and Lorentz materials, preview n/k and run dispersive GPU fields',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="materials"]').click();
 const dialog=page.locator('.material-dialog');
 await dialog.getByRole('button',{name:'+ Add material',exact:true}).click();
 await dialog.getByLabel('Material name',{exact:true}).fill('Air');await dialog.getByLabel('Material name',{exact:true}).press('Tab');
 await expect(dialog.locator('.material-status')).toContainText('unique');
 await dialog.getByLabel('Material model',{exact:true}).selectOption('drude');
 await dialog.getByLabel('Material name',{exact:true}).fill('Drude test');await dialog.getByLabel('Material name',{exact:true}).press('Tab');
 await dialog.getByLabel('Permittivity (epsilon infinity)',{exact:true}).fill('2');await dialog.getByLabel('Permittivity (epsilon infinity)',{exact:true}).press('Tab');
 await dialog.getByRole('button',{name:'Plot n / k',exact:true}).click();
 await expect(dialog.locator('.material-status')).toContainText('301 wavelengths');
 await dialog.getByLabel('Material wavelength stop',{exact:true}).fill('1.9');await dialog.getByLabel('Material wavelength stop',{exact:true}).press('Tab');
 await expect(dialog.locator('.material-status')).toContainText('Parameters changed');
 await dialog.getByRole('button',{name:'Plot n / k',exact:true}).click();
 await expect(dialog.locator('.material-status')).toContainText('301 wavelengths');
 await page.screenshot({path:'results/ui-material-drude.png',fullPage:true});
 await dialog.getByRole('button',{name:'Apply materials',exact:true}).click();
 await expect(dialog).not.toBeVisible();
 await page.locator('#tree .tree-row').nth(1).click();
 await page.getByLabel('material',{exact:true}).selectOption('Drude test');
 await page.locator('[data-action="materials"]').click();
 await dialog.getByLabel('Material list',{exact:true}).selectOption({label:'Drude test'});
 await dialog.getByLabel('Material model',{exact:true}).selectOption('lorentz');
 await dialog.getByLabel('Lorentz linewidth',{exact:true}).fill('200000000000000');await dialog.getByLabel('Lorentz linewidth',{exact:true}).press('Tab');
 await dialog.getByRole('button',{name:'Plot n / k',exact:true}).click();
 await expect(dialog.locator('.material-status')).toContainText('Positive k means absorption');
 await page.screenshot({path:'results/ui-material-lorentz.png',fullPage:true});
 await dialog.getByRole('button',{name:'Apply materials',exact:true}).click();
 await page.locator('[data-select="fdtd"]').click();
 await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');
 await page.getByLabel('time steps',{exact:true}).fill('400');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await page.locator('#run-button').click();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:60000});
 if(process.env.TORCHFDTD_TEST_CUDA)await expect(page.locator('.run-summary')).toContainText('CUDA graph');
 const save=page.waitForEvent('download');await page.locator('[data-action="save"]').click();
 const project=JSON.parse(fs.readFileSync(await (await save).path(),'utf8'));
 expect(project.materials.find(m=>m.name==='Drude test').linewidth_rad_s).toBe(2e14);
 expect(errors).toEqual([]);
});

test('independently import a real Lorentz FSP and execute on GPU',async({page})=>{
 test.skip(!process.env.TORCHFDTD_MATERIAL_FSP,'Requires controlled material fixture');
 test.setTimeout(120000);
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="fsp-native"]').click();
 if(process.env.TORCHFDTD_DRUDE_FSP){
  await page.locator('#fsp-native-input').setInputFiles(process.env.TORCHFDTD_DRUDE_FSP);
  await expect(page.locator('.native-issues')).toContainText('Experimental curved Drude geometry',{timeout:20000});
 }
 await page.locator('#fsp-native-input').setInputFiles(process.env.TORCHFDTD_MATERIAL_FSP);
 await expect(page.locator('#fsp-native-status')).toContainText('Ready to open',{timeout:20000});
 await expect(page.locator('.native-issues')).toContainText('trapezoidal ADE');
 await page.locator('[data-native="load"]').click();
 await page.locator('[data-action="materials"]').click();
 await expect(page.getByLabel('Material model',{exact:true})).toHaveValue('lorentz');
 await expect(page.getByLabel('Lorentz resonance',{exact:true})).toHaveValue('1700000000000000');
 await page.locator('.material-dialog').getByRole('button',{name:'Close',exact:true}).click();
 await page.locator('#run-button').click();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await expect(page.locator('.run-summary')).toContainText('CUDA graph');
});
