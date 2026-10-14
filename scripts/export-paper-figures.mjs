import {execFileSync} from 'node:child_process';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
const require=createRequire(new URL('../website/package.json',import.meta.url));
const sharp=require('sharp');
const pdf=process.argv[2];
if (!pdf) throw new Error('Pass the source manuscript PDF');
const output=fileURLToPath(new URL('../website/public/assets/paper/',import.meta.url));
const cache=fileURLToPath(new URL('../.local/paper/',import.meta.url));
await mkdir(output,{recursive:true}); await mkdir(cache,{recursive:true});
// Coordinates refer to the full PDF page rendered with its long edge at 4800px.
const figures=[
  {id:'motivation',page:2,figure:2,title:'Motivation',crop:{left:648,top:432,width:2416,height:1112}},
  {id:'pipeline',page:5,figure:4,title:'Pipeline',crop:{left:656,top:2224,width:2408,height:952}},
];
for (const f of figures) {
  const prefix=join(cache,f.id);
  execFileSync('pdftoppm',['-f',String(f.page),'-l',String(f.page),'-scale-to','4800','-png','-singlefile',pdf,prefix]);
  await sharp(prefix+'.png').extract(f.crop).webp({quality:94}).toFile(join(output,f.id+'.webp'));
  await sharp(prefix+'.png').extract(f.crop).resize({width:1440}).webp({quality:90}).toFile(join(output,f.id+'-1440.webp'));
}
const manifest={source:'Nereus manuscript supplied by the authors',sha256:createHash('sha256').update(await readFile(pdf)).digest('hex'),
  method:'Original PDF rendering; page margins and surrounding text removed. Figure contents and proportions unchanged. The manuscript PDF is not distributed.',
  figures};
await writeFile(join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(figures);
