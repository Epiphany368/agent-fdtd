import {test,expect} from '@playwright/test';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';

// G8-03: one CPU-only journey from a GDS import to the data and GDS exports. Every
// step asserts visible state; page errors and console errors fail the test.
test('workbench journey: GDS import, materials, source, boundaries, mesh preview, preflight, queue, cancel, results, exports',async({page},testInfo)=>{
 test.setTimeout(240000);
 const started=Date.now(),timing={};
 const lap=name=>{timing[name]=Date.now()-started;};
 const python=process.env.TORCHFDTD_TEST_PYTHON||'python';
 const file=testInfo.outputPath('journey.gds');
 execFileSync(python,['-c',`import gdstk,tempfile,pathlib,sys
with tempfile.TemporaryDirectory(prefix='.gds-ui-',dir='.') as d:
 p=pathlib.Path(d)/'fixture.gds'
 lib=gdstk.Library();cell=lib.new_cell('JOURNEY');cell.add(gdstk.rectangle((-1.5,-.25),(1.5,.25),layer=1,datatype=0));lib.write_gds(p);pathlib.Path(sys.argv[1]).write_bytes(p.read_bytes())`,file]);
 const errors=[],consoleErrors=[],downloads=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text());});page.on('download',d=>downloads.push(d));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await expect(page.locator('#connection')).not.toContainText('Connecting');

 // 1. CAD/GDS import replacing the example geometry, with an explicit Z stack and material per layer.
 await page.locator('[data-action="gds"]').click();await page.locator('#gds-input').setInputFiles(file);
 await expect(page.locator('#gds-cell')).toHaveValue('JOURNEY');
 await page.getByLabel('Include 1/0',{exact:true}).check();
 await page.getByLabel('Z min 1/0',{exact:true}).fill('-0.2');await page.getByLabel('Z max 1/0',{exact:true}).fill('0.2');
 const materialSelect=page.getByLabel('Material 1/0',{exact:true}),material=await materialSelect.locator('option').nth(1).getAttribute('value');
 await materialSelect.selectOption(material);await page.locator('#gds-replace').check();
 await page.locator('[data-gds="preview"]').click();await expect(page.locator('#gds-status')).toContainText('Ready to apply');
 await page.locator('[data-gds="apply"]').click();await expect(page.locator('#gds-dialog')).not.toBeVisible();
 await expect(page.locator('#tree')).not.toContainText('waveguide');
 const imported=page.locator('#tree .tree-row').filter({hasText:'JOURNEY'});await expect(imported).toHaveCount(1);
 lap('gds_import');

 // 2. Material assignment is visible on the imported structure; the mesh order is edited.
 await imported.click();
 await expect(page.locator('#property-type')).toHaveText('polygon');
 await expect(page.getByLabel('material',{exact:true})).toHaveValue(material);
 await expect(page.locator('#properties')).toContainText('refractive index');
 await page.getByLabel('mesh order',{exact:true}).fill('1');await page.getByLabel('mesh order',{exact:true}).press('Tab');
 await expect(page.getByLabel('mesh order',{exact:true})).toHaveValue('1');
 lap('materials');

 // 3. A sheet source placed left of the structure and a time monitor to its right.
 await page.locator('[data-add="plane"]').click();
 await expect(page.locator('#property-type')).toHaveText('plane');
 await page.getByLabel('x',{exact:true}).fill('-2.5');await page.getByLabel('x',{exact:true}).press('Tab');
 await page.getByLabel('y span',{exact:true}).fill('2');await page.getByLabel('y span',{exact:true}).press('Tab');
 await expect(page.getByLabel('x',{exact:true})).toHaveValue('-2.5');
 await page.locator('[data-add="monitor"]').click();
 await page.getByLabel('x',{exact:true}).fill('2.5');await page.getByLabel('x',{exact:true}).press('Tab');
 await expect(page.locator('#tree .tree-group').filter({hasText:'Sources'})).toContainText('2');
 await expect(page.locator('#tree .tree-group').filter({hasText:'Monitors'})).toContainText('3');
 lap('source_monitor');

 // 4. Boundaries: PEC walls on y and a thicker PML on x, on the CPU.
 await page.locator('[data-select="fdtd"]').click();
 await page.getByLabel('y min bc',{exact:true}).selectOption('pec');await page.getByLabel('y max bc',{exact:true}).selectOption('pec');
 await page.getByLabel('PML layers',{exact:true}).fill('10');await page.getByLabel('PML layers',{exact:true}).press('Tab');
 await expect(page.getByLabel('y min bc',{exact:true})).toHaveValue('pec');await expect(page.getByLabel('y max bc',{exact:true})).toHaveValue('pec');
 await expect(page.getByLabel('PML layers',{exact:true})).toHaveValue('10');
 await page.getByLabel('resource',{exact:true}).selectOption('cpu');
 await expect(page.locator('#mesh-summary strong')).toContainText('×');
 lap('boundaries');

 // 5. The mesh preview reports the same cell counts as the summary card and the server.
 const previewResponse=page.waitForResponse(r=>r.url().endsWith('/api/mesh/preview'));
 await page.getByRole('button',{name:'Preview simulation mesh',exact:true}).click();
 const preview=(await (await previewResponse).json()).summary;
 const modal=page.locator('.mesh-dialog');await expect(modal).toBeVisible();
 await expect(modal.locator('.mesh-metrics strong')).toHaveText(`${preview.shape.join(' × ')} cells`);
 await expect(page.locator('#mesh-summary strong')).toHaveText(preview.shape.join(' × '));
 await expect(page.locator('#mesh-summary')).toContainText(`${preview.cells.toLocaleString('en-US')} cells`);
 expect(preview.cells).toBe(preview.shape.reduce((a,b)=>a*b,1));
 await expect(modal.locator('canvas')).toBeVisible();
 await modal.getByRole('button',{name:'Close',exact:true}).click();await expect(modal).not.toBeVisible();
 lap('mesh_preview');

 // 6. Resource preflight: /api/validate resolves the execution mode shown in the status line.
 const validation=page.waitForResponse(r=>r.url().endsWith('/api/validate'));
 await page.getByLabel('time steps',{exact:true}).fill('20000');await page.getByLabel('time steps',{exact:true}).press('Tab');
 const preflight=await (await validation).json();
 expect(preflight.execution.mode).toBe('resident');expect(preflight.execution.backend).toBe('cpu');
 expect(preflight.plan_hash).toMatch(/^[0-9a-f]{64}$/);
 await expect(page.locator('#execution-status')).toContainText('Auto → Resident on CPU');
 await expect(page.locator('#execution-status')).toContainText(preflight.execution.reason.slice(0,40));
 lap('preflight');

 // 7. Submit to the queue, cancel the running job, then resubmit a short run and wait for completion.
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const first=await (await submitted).json();expect(first.status).toBe('queued');
 await expect(page.locator('#mode-badge')).toHaveText('RUNNING');
 await expect(page.locator('#stop-button')).toBeEnabled();
 await expect(page.locator('#status-text')).toContainText(/Queued on solver|Calculating fields/);
 await page.locator('#stop-button').click();
 await expect(page.locator('#messages')).toContainText('Stop requested');
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:120000});
 await expect(page.locator('.run-summary')).toContainText('Cancelled');
 const cancelled=await (await page.request.get('/api/jobs/'+first.id)).json();
 expect(cancelled.status).toBe('cancelled');expect(cancelled.summary.cancelled).toBe(true);expect(cancelled.summary.steps).toBeLessThan(20000);
 expect(cancelled.plan_hash).toBe(preflight.plan_hash);
 lap('cancel');
 await page.locator('#layout-button').click();await expect(page.locator('#mode-badge')).toHaveText('LAYOUT');
 await page.locator('[data-select="fdtd"]').click();
 await page.getByLabel('time steps',{exact:true}).fill('400');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await expect(page.locator('#stale-banner')).toBeVisible();
 const resubmitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const second=await (await resubmitted).json();expect(second.id).not.toBe(first.id);
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:120000});
 await expect(page.locator('.run-summary')).toContainText('Step limit reached');
 await expect(page.locator('.run-summary')).toContainText('400 / 400 steps');
 await expect(page.locator('#stale-banner')).toBeHidden();
 lap('rerun');

 // 8. The results overlay renders the field snapshot and lists the monitors.
 await expect(page.locator('#results-tree')).toContainText('field snapshots');
 await expect(page.locator('#results-tree')).toContainText('input');
 await expect(page.locator('#results-tree')).toContainText('monitor_');
 await expect(page.locator('#frame-label')).toHaveText('Step 400');
 await expect(page.locator('#field-label')).toContainText('Ez');
 await expect(page.locator('#field-label')).not.toContainText('STALE');
 const pixels=await page.evaluate(()=>{const c=document.querySelector('#field-canvas'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data,colors=new Set();for(let i=0;i<d.length;i+=4*97)colors.add((d[i]<<16)|(d[i+1]<<8)|d[i+2]);return {width:c.width,height:c.height,colors:colors.size};});
 expect(pixels.width).toBeGreaterThan(100);expect(pixels.colors).toBeGreaterThan(20);
 await page.locator('[data-plot="spectrum"]').click();await expect(page.locator('#plot-axis')).toBeVisible();
 await page.screenshot({path:testInfo.outputPath('journey-results.png'),fullPage:true});
 lap('results_overlay');

 // 9. Exports: the NPZ result, the project JSON with its version fields, and the GDS with its stack sidecar.
 await page.locator('[data-action="download"]').click();
 await expect.poll(()=>downloads.length).toBe(1);
 expect(downloads[0].suggestedFilename()).toMatch(/^torchfdtd-[0-9a-f]{8}\.npz$/);
 const npz=fs.readFileSync(await downloads[0].path());expect(npz.subarray(0,2).toString('latin1')).toBe('PK');expect(npz.length).toBeGreaterThan(1000);
 await page.locator('[data-action="save"]').click();
 await expect.poll(()=>downloads.length).toBe(2);
 const project=JSON.parse(fs.readFileSync(await downloads[1].path(),'utf8'));
 expect(project.revision).toBeGreaterThan(0);expect(project.content_sha256).toMatch(/^[0-9a-f]{64}$/);
 expect(project.structures[0].kind).toBe('polygon');expect(project.region.boundaries.y_min.kind).toBe('pec');
 const check=await (await page.request.post('/api/validate',{data:project})).json();
 expect(check.stored_content_sha256_matches).toBe(true);
 const completed=await (await page.request.get('/api/jobs/'+second.id)).json();expect(check.plan_hash).toBe(completed.plan_hash);expect(completed.plan_hash).not.toBe(preflight.plan_hash);
 await page.locator('[data-action="export-gds"]').click();
 const exportDialog=page.locator('#gds-export-dialog');await expect(exportDialog).toBeVisible();
 await expect(exportDialog).toContainText('1 structure ready');
 await exportDialog.getByLabel('GDS cell name',{exact:true}).fill('EXPORTED');
 await exportDialog.locator('[data-gds-export="run"]').click();
 await expect(exportDialog.locator('#gds-export-status')).toContainText('Exported 1 structure');
 await expect.poll(()=>downloads.length).toBe(4);
 const gds=downloads.find(d=>d.suggestedFilename().endsWith('.gds')),sidecar=downloads.find(d=>d.suggestedFilename().endsWith('.gds.json'));
 const gdsBytes=fs.readFileSync(await gds.path());expect([...gdsBytes.subarray(0,4)]).toEqual([0,6,0,2]);
 const stack=JSON.parse(fs.readFileSync(await sidecar.path(),'utf8'));
 expect(stack.cell).toBe('EXPORTED');expect(stack.structures).toBe(1);expect(stack.layer_stack[0]).toMatchObject({layer:1,datatype:0,material,z_min:-0.2,z_max:0.2});
 await exportDialog.getByRole('button',{name:'Close',exact:true}).click();
 lap('exports');

 timing.total_ms=Date.now()-started;
 fs.writeFileSync(testInfo.outputPath('journey-timing.json'),JSON.stringify(timing,null,2));
 await testInfo.attach('journey-timing',{body:JSON.stringify(timing),contentType:'application/json'});
 expect(errors).toEqual([]);expect(consoleErrors).toEqual([]);
});
