import {test,expect} from '@playwright/test';
import fs from 'node:fs';

// G6-01: the materials dialog records the source, licence and raw hash of an imported table, shows the
// fitted band and the discretization n/k error, surfaces the validation's extrapolation warning for the
// project sources, and the saved project keeps the provenance.
test('imported optical table keeps its provenance and the band panel reports extrapolation',async({page})=>{
 test.setTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const csv=fs.readFileSync('tests/fixtures/materials/sio2_sellmeier_malitson1965.csv','utf8');
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="materials"]').click();const dialog=page.locator('.material-dialog');
 await dialog.getByRole('button',{name:'+ Add material',exact:true}).click();
 await dialog.getByLabel('Material name',{exact:true}).fill('SiO2 Malitson');await dialog.getByLabel('Material name',{exact:true}).press('Tab');
 await expect(dialog.locator('[data-provenance]')).toContainText('No provenance recorded');
 await dialog.locator('.material-fit summary').click();
 await dialog.getByLabel('Optical data file',{exact:true}).setInputFiles({name:'sio2_sellmeier_malitson1965.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
 await expect(dialog.getByLabel('Optical data table',{exact:true})).toHaveValue(/wavelength_um/);
 await expect(dialog.getByLabel('Optical data reference',{exact:true})).toHaveValue('sio2_sellmeier_malitson1965.csv');
 await dialog.getByLabel('Optical data reference',{exact:true}).fill('Malitson 1965 Sellmeier (fused silica)');
 await dialog.getByLabel('Optical data licence',{exact:true}).fill('formula-generated test table');
 const hashed=page.waitForResponse(r=>r.url().endsWith('/api/materials/provenance'));
 await dialog.getByRole('button',{name:'Import data',exact:true}).click();
 const provenance=await (await hashed).json();
 expect(provenance.source).toBe('Malitson 1965 Sellmeier (fused silica)');expect(provenance.file_name).toBe('sio2_sellmeier_malitson1965.csv');
 expect(provenance.raw_sha256).toMatch(/^[0-9a-f]{64}$/);
 await expect(dialog.locator('[data-data-status]')).toContainText('81 samples');
 await dialog.getByLabel('Maximum fit poles',{exact:true}).fill('3');
 await dialog.getByLabel('Fit tolerance',{exact:true}).fill('0.0001');
 await dialog.getByRole('button',{name:'Fit optical data',exact:true}).click();
 await expect(dialog.locator('.fit-status')).toContainText('Tolerance met',{timeout:30000});
 await dialog.getByRole('button',{name:'Use fitted material',exact:true}).click();
 const panel=dialog.locator('[data-provenance]');
 await expect(panel).toContainText('Malitson 1965 Sellmeier (fused silica)');
 await expect(panel).toContainText('formula-generated test table');
 await expect(panel).toContainText(provenance.raw_sha256.slice(0,16));
 await expect(panel).toContainText('0.4–2 µm');
 // The waveguide example source (1.55 um) lies inside the fitted band.
 await expect(panel.locator('[data-band-status]')).toContainText('inside the fitted band');
 await dialog.getByRole('button',{name:'Plot n / k',exact:true}).click();
 await expect(panel.locator('[data-discretization]')).toContainText('max |Δn|');
 await expect(panel.locator('[data-discretization]')).toContainText('trapezoidal ADE');
 await page.screenshot({path:'results/ui-material-provenance.png',fullPage:true});
 await dialog.getByRole('button',{name:'Apply materials',exact:true}).click();
 await expect(dialog).not.toBeVisible();
 await page.locator('#tree .tree-row').nth(1).click();await page.getByLabel('material',{exact:true}).selectOption('SiO2 Malitson');
 // Move the source beyond the fitted band: the validation warning appears in the panel.
 await page.locator('[data-select="source"]').click();
 await page.getByLabel('wavelength',{exact:true}).fill('2.5');await page.getByLabel('wavelength',{exact:true}).press('Tab');
 await page.locator('[data-action="materials"]').click();
 await dialog.getByLabel('Material list',{exact:true}).selectOption({label:'SiO2 Malitson'});
 await expect(panel.locator('[data-band-status]')).toContainText('lies outside the SiO2 Malitson fit band');
 await expect(panel.locator('[data-band-status]')).toHaveClass(/band-warning/);
 await dialog.getByRole('button',{name:'Cancel',exact:true}).click();
 const saved=page.waitForEvent('download');await page.locator('[data-action="save"]').click();
 const project=JSON.parse(fs.readFileSync(await (await saved).path(),'utf8'));
 const material=project.materials.find(m=>m.name==='SiO2 Malitson');
 expect(material.provenance.raw_sha256).toBe(provenance.raw_sha256);
 expect(material.provenance.licence).toBe('formula-generated test table');
 expect(material.fit_band_um[0]).toBeCloseTo(.4,12);expect(material.fit_band_um[1]).toBeCloseTo(2,12);
 expect(material.samples.wavelength_um).toHaveLength(81);
 expect(errors).toEqual([]);
});
