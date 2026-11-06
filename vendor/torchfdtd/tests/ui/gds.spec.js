import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';

test('GDS explicit stack preview, native CAD and project/Python exports',async({page},testInfo)=>{
 const python=process.env.TORCHFDTD_TEST_PYTHON||'python';
 const file=testInfo.outputPath('independent.gds');
 execFileSync(python,['-c',`import gdstk,tempfile,pathlib,sys
with tempfile.TemporaryDirectory(prefix='.gds-ui-',dir='.') as d:
 p=pathlib.Path(d)/'fixture.gds'
 lib=gdstk.Library();cell=lib.new_cell('UI_TOP');cell.add(gdstk.rectangle((-.4,-.2),(.4,.2),layer=17,datatype=3),gdstk.rectangle((-.1,-.05),(.1,.05),layer=18));lib.write_gds(p);pathlib.Path(sys.argv[1]).write_bytes(p.read_bytes())`,file]);
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="gds"]').click();await page.locator('#gds-input').setInputFiles(file);
 await expect(page.locator('#gds-cell')).toHaveValue('UI_TOP');
 await page.getByLabel('Include 17/3',{exact:true}).check();
 await page.getByLabel('Z min 17/3',{exact:true}).fill('-.1');await page.getByLabel('Z max 17/3',{exact:true}).fill('.1');
 const material=await page.getByLabel('Material 17/3',{exact:true}).locator('option').nth(1).getAttribute('value');
 await page.getByLabel('Material 17/3',{exact:true}).selectOption(material);
 await page.locator('[data-gds="preview"]').click();await expect(page.locator('#gds-status')).toContainText('Unmapped');
 await page.getByLabel('Etch by 17/3',{exact:true}).fill('18/x');
 await page.locator('[data-gds="preview"]').click();await expect(page.locator('#gds-status')).toContainText('Etch pairs');
 await page.getByLabel('Etch by 17/3',{exact:true}).fill('18/0');
 await page.locator('[data-gds="preview"]').click();await expect(page.locator('#gds-status')).toContainText('Ready to apply');
 const reportDownload=page.waitForEvent('download');await page.locator('[data-gds="report"]').click();
 const report=JSON.parse(fs.readFileSync(await(await reportDownload).path(),'utf8'));expect(report.cell).toBe('UI_TOP');
 expect(report.hole_count).toBe(1);expect(report.etched_layers).toEqual([{layer:17,datatype:3,etch_by:[[18,0]],polygons:1}]);
 await page.locator('[data-gds="apply"]').click();await expect(page.locator('#gds-dialog')).not.toBeVisible();
 const saved=page.waitForEvent('download');await page.locator('[data-action="save"]').click();
 const project=JSON.parse(fs.readFileSync(await(await saved).path(),'utf8'));
 const shape=project.structures.find(s=>s.kind==='polygon');expect(shape).toBeTruthy();expect(shape.material).toBe(material);expect(shape.size).toEqual([.8,.4,.2]);
 expect(shape.holes).toHaveLength(1);expect(shape.holes[0]).toHaveLength(4);
 await expect(page.locator('#tree')).toContainText(shape.name);await expect(page.locator('canvas').first()).toBeVisible();
 // The native editor shows the hole as its own contour block and the CAD views render it.
 await page.locator(`[data-select="${shape.id}"]`).click();
 await expect(page.locator('#properties')).toContainText('1 hole contour');
 await page.getByRole('button',{name:'Edit polygon vertices',exact:true}).click();
 const editor=page.locator('.geometry-dialog').getByLabel('Polygon vertices',{exact:true});
 const blocks=(await editor.inputValue()).split(/\n\s*\n/);expect(blocks).toHaveLength(2);expect(blocks[1].trim().split('\n')).toHaveLength(4);
 await page.locator('.geometry-dialog').getByRole('button',{name:'Close',exact:true}).click();
 await page.screenshot({path:testInfo.outputPath('gds-holed-cad.png'),fullPage:true});
 await page.locator('[data-action="python"]').click();await expect(page.locator('#python-editor')).toHaveValue(/polygon/);
 const pyDownload=page.waitForEvent('download');await page.locator('[data-action="export-python"]').click();
 expect(fs.readFileSync(await(await pyDownload).path(),'utf8')).toContain(shape.id);
 await page.screenshot({path:testInfo.outputPath('gds-native-cad.png'),fullPage:true});expect(errors).toEqual([]);
});
