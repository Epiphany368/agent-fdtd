import {test,expect} from '@playwright/test';

// G6-02: the source dialog previews what the solver realizes (spatial amplitude and Bloch phase, polarization,
// effective bandwidth, fixed-k_parallel incidence with its angle range) and refuses a fixed-angle request with
// the registry message instead of drawing it.
test('Bloch sheet preview states fixed k_parallel and refuses a fixed-angle definition',async({page})=>{
 const p=await (await page.request.get('/api/examples/scatterer')).json();
 p.structures=[];p.region.size=[8,1.2,1];p.region.mesh=.05;p.region.steps=800;p.region.pml_cells=10;p.region.backend='cpu';
 p.region.boundaries.y_min.kind='bloch';p.region.boundaries.y_max.kind='bloch';
 p.region.bloch_phase=[0,2*Math.PI/1.55*Math.sin(20*Math.PI/180)*1.2,0];
 p.sources[0].center=[-2.5,0,0];p.sources[0].size=[0,1.2,0];p.sources[0].pulse_cycles=3;p.sources[0].injection='soft';
 // The init script runs on every load; the second part of the test rewrites the stored project itself.
 await page.addInitScript(p=>{if(!localStorage.getItem('torchfdtd.spec.seeded')){localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p));localStorage.setItem('torchfdtd.spec.seeded','1');}},p);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await page.locator('[data-select="source"]').click();
 const pending=page.waitForResponse(r=>r.url().includes('/sources/')&&r.url().endsWith('/preview'));
 await page.getByRole('button',{name:'Preview time signal / spectrum',exact:true}).click();
 const preview=await (await pending).json();
 expect(preview.incidence.kind).toBe('fixed_k_parallel');
 expect(preview.incidence.angle_deg.carrier.angle_deg).toBeCloseTo(20,6);
 expect(preview.bandwidth.threshold_fraction).toBe(.01);
 expect(preview.spatial[0].phase_rad_by_axis.y).toHaveLength(24);
 const dialog=page.locator('.source-dialog');
 await expect(dialog.locator('[data-preview-incidence]')).toContainText('Fixed k∥ (Bloch phase)');
 await expect(dialog.locator('[data-preview-incidence]')).toContainText('20.000°');
 await expect(dialog.locator('[data-preview-incidence]')).toContainText('varies across the band');
 await expect(dialog.locator('[data-preview-bandwidth]')).toContainText('Effective bandwidth');
 await expect(dialog.locator('[data-preview-bandwidth]')).toContainText('0.01 of its peak');
 await expect(dialog.locator('[data-preview-polarization]')).toContainText('electric vector (0.0000, 0.0000, 1.0000)');
 await expect(dialog.locator('[data-preview-spatial]')).toContainText('Ez soft sheet');
 await expect(dialog.locator('[data-preview-spatial]')).toContainText('Bloch spatial phase');
 await dialog.getByRole('button',{name:'Spatial phase',exact:true}).click();
 await page.screenshot({path:'results/ui-source-preview-phase.png',fullPage:true});
 const refused=page.waitForResponse(r=>r.url().includes('incidence=fixed_angle'));
 await page.getByLabel('Incidence definition',{exact:true}).selectOption('fixed_angle');
 expect((await refused).status()).toBe(422);
 await expect(dialog.locator('[data-preview-incidence]')).toContainText('Refused: Fixed-angle broadband injection is not implemented');
 await expect(dialog.getByRole('alert')).toContainText('feature inventory source.angle and boundary.bfast are missing');
 await page.screenshot({path:'results/ui-source-preview-refused.png',fullPage:true});
 await dialog.getByRole('button',{name:'Close',exact:true}).click();
 // A normal-incidence sheet on periodic boundaries.
 await page.evaluate(()=>{const p=JSON.parse(localStorage.getItem('torchfdtd.project.v1'));p.region.boundaries.y_min.kind='periodic';p.region.boundaries.y_max.kind='periodic';p.region.bloch_phase=[0,0,0];localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p));});
 await page.reload();await page.locator('[data-select="source"]').click();
 await page.getByRole('button',{name:'Preview time signal / spectrum',exact:true}).click();
 await expect(dialog.locator('[data-preview-incidence]')).toContainText('Normal incidence');
 await expect(dialog.getByRole('button',{name:'Spatial phase',exact:true})).toHaveCount(0);
 expect(errors).toEqual([]);
});
