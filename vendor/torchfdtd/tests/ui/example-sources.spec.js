import {test,expect} from '@playwright/test';

// The cylinder example launches a one-way sheet toward +x; the Project schema has no eigenmode source,
// so the waveguide is excited by a soft sheet across its cross-section (bidirectional, double arrow).
test('the example sheets are a soft cross-section sheet in the guide and a one-way sheet toward +x, and the views draw their arrows',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const waveguide=await (await page.request.get('/api/examples/waveguide')).json();
 expect(waveguide.sources[0]).toMatchObject({id:'source',kind:'plane',injection:'soft',center:[-2.5,0,0],size:[0,1,0]});
 expect(waveguide.region.boundaries.y_min.kind).toBe('pml');
 expect(waveguide.structures[0].size).toEqual([8,0.65,0.4]);
 expect(waveguide.monitors.map(m=>m.id)).toEqual(['input','output']);
 const scatterer=await (await page.request.get('/api/examples/scatterer')).json();
 expect(scatterer.sources[0]).toMatchObject({id:'source',kind:'plane',injection:'oneway',normal:'x',direction:'+'});
 expect(scatterer.region.boundaries.y_min.kind).toBe('periodic');
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="scatterer"]').click();
 await page.locator('[data-select="source"]').click();
 await expect(page.locator('#property-type')).toHaveText('plane');
 await expect(page.getByLabel('injection',{exact:true})).toHaveValue('oneway');
 await expect(page.getByLabel('direction',{exact:true})).toHaveValue('+');
 await expect(page.getByLabel('propagation axis',{exact:true})).toHaveValue('x');
 await page.locator('[data-example="waveguide"]').click();
 await page.locator('[data-select="source"]').click();
 await expect(page.getByLabel('injection',{exact:true})).toHaveValue('soft');
 await expect(page.getByLabel('y span',{exact:true})).toHaveValue('1');
 await page.screenshot({path:'results/ui-example-sources.png',fullPage:true});
 expect(errors).toEqual([]);
});

test('the region panel and the visualizer report stored frames per optical period and warn when they alias',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 const rate=page.locator('#snapshot-rate');
 await expect(rate).toHaveText('4.4');
 await expect(page.locator('#snapshot-status')).toContainText('at most 100 frames are stored');
 await expect(page.locator('#snapshot-warning')).toHaveCount(0);
 await expect(page.locator('#mesh-summary')).not.toContainText('alias the carrier');
 await page.getByLabel('snapshot every',{exact:true}).fill('40');await page.getByLabel('snapshot every',{exact:true}).press('Tab');
 await expect(rate).toHaveText('1.1');
 await expect(page.locator('#snapshot-warning')).toContainText('the playback will look like backward motion');
 await expect(page.locator('#mesh-summary')).toContainText('alias the carrier');
 const gpu=page.getByLabel('GPU',{exact:true});
 if(await gpu.isEnabled())await gpu.uncheck();
 await page.getByLabel('time steps',{exact:true}).fill('200');await page.getByLabel('time steps',{exact:true}).press('Tab');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();await submitted;
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await expect(page.locator('#frame-label')).toContainText('aliased: 1.1 frames per optical period');
 await page.locator('#layout-button').click();
 await page.getByLabel('snapshot every',{exact:true}).fill('5');await page.getByLabel('snapshot every',{exact:true}).press('Tab');
 await expect(page.locator('#snapshot-warning')).toHaveCount(0);
 expect(errors).toEqual([]);
});
