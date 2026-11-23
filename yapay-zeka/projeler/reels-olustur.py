import subprocess, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
W,H,FPS=1080,1920,30
B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; R="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_fc={}
def F(p,s):
    if (p,s) not in _fc: _fc[(p,s)]=ImageFont.truetype(p,s)
    return _fc[(p,s)]
Y=(255,205,60)
# (duration, images, title, subtitle, accent, kind)  kind: hook/title/course/list/cta
S=[(2.6,["resim2"],"Bu yaz\nne\nyapacaksın?","",Y,"hook"),
   (2.6,["grup5"],"GSB GENÇLİK\nYAZ KULÜBÜ\n2026","Denizli Gençlik Merkezi",(255,255,255),"title"),
   (2.6,["robot"],"Temel Robotik\nKodlama","Geleceği sen kodla",(130,255,170),"course"),
   (2.6,["satranc"],"Akıl Zekâ\nOyunları","Hamle sırası sende",(255,230,120),"course"),
   (2.6,["office"],"Office\nWord","Bilgisayarda ustalaş",(120,200,255),"course"),
   (3.2,["keman2","muzik"],"Keman\nGitar\nPiyano","Sahne seni bekliyor",(120,220,255),"course"),
   (2.6,["resim3"],"Resim","Fırçanı al, gel!",(255,120,160),"course"),
   (2.6,["ebru"],"El\nSanatları","Hayal et, üret",(255,170,80),"course"),
   (2.6,["hat"],"Osmanlıca","Geçmişi oku",(230,200,140),"course"),
   (2.6,["ingilizce"],"İngilizce\n&\nDiksiyon","Kendini ifade et",(160,230,255),"course"),
   (4.6,["kod"],"","",Y,"list"),
   (5.2,["grup5"],"","",Y,"cta")]
COURSES=["Temel Robotik Kodlama","Akıl Zeka Oyunları","Office Word","El Sanatları","Osmanlıca","İngilizce","Diksiyon",
         "Değerler Eğitimi","Keman","Gitar","Piyano","Resim","Drama","Kur'an Kursu"]
t0=0; SC=[]
for s in S: SC.append((t0,t0+s[0])+s[1:]); t0+=s[0]
TOTAL=t0
def cover(name,scale=1.18):
    im=Image.open(f"img/{name}.jpg").convert("RGB"); tw,th=int(W*scale),int(H*scale)
    r=max(tw/im.width,th/im.height); return im.resize((int(im.width*r)+1,int(im.height*r)+1),Image.LANCZOS)
imgs={n:cover(n) for s in S for n in s[1]}
grad=Image.new("RGBA",(W,H)); gd=ImageDraw.Draw(grad)
for y in range(H):
    a=int(max(0,(y-H*0.35)/(H*0.65))**1.4*220)+(int((1-y/(H*0.18))*120) if y<H*0.18 else 0)
    gd.line((0,y,W,y),fill=(0,0,0,min(255,a)))
ease=lambda t: 1-(1-max(0,min(1,t)))**3
blurcache={}
p=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
  "-f","lavfi","-i","anullsrc=r=44100:cl=stereo","-shortest","-c:v","libx264","-preset","medium","-crf","19",
  "-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart",sys.argv[1]],stdin=subprocess.PIPE)
foot=F(B,34); nframes=int(round(TOTAL*FPS))
def ctext(d,txt,font,y,fill): w=d.textlength(txt,font=font); d.text(((W-w)/2,y),txt,font=font,fill=fill)
for i in range(nframes):
    t=i/FPS
    k=next((j for j,s in enumerate(SC) if s[0]<=t<s[1]),len(SC)-1)
    st,en,names,title,sub,acc,kind=SC[k]; lt=t-st; dur=en-st
    seg=dur/len(names); ni=min(len(names)-1,int(lt/seg)); name=names[ni]; u=(lt-ni*seg)/seg
    zd=1 if k%2==0 else -1
    src=imgs[name]; z=1.0+0.15*(u if zd>0 else 1-u)
    cw,ch=W*1.18/z,H*1.18/z
    cx=(src.width-cw)*(0.5+0.08*(u-0.5)*zd); cx=max(0,min(src.width-cw,cx)); cy=(src.height-ch)/2
    fr=src.crop((int(cx),int(cy),int(cx+cw),int(cy+ch))).resize((W,H),Image.BILINEAR)
    if kind in ("title","list","cta"):
        fr=fr.filter(ImageFilter.GaussianBlur(18)); fr=ImageEnhance.Brightness(fr).enhance(0.42)
    elif kind=="hook": fr=ImageEnhance.Brightness(fr).enhance(0.6)
    fr=Image.alpha_composite(ImageEnhance.Color(fr).enhance(1.15).convert("RGBA"),grad)
    L=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    if title:
        lines=title.split("\n"); sz=170 if kind=="course" else 150
        while max(d.textlength(l,font=F(B,sz)) for l in lines)>W-140: sz-=4
        tf=F(B,sz); lh=int(sz*1.08); center=kind in ("hook","title")
        y0=(H//2-len(lines)*lh//2-80) if center else H-560-len(lines)*lh
        for n,l in enumerate(lines):
            la=ease((lt-0.05-n*0.1)/0.35)
            x=(W-d.textlength(l,font=tf))/2 if center else 70-int(120*(1-la))
            yy=y0+n*lh+(int(50*(1-la)) if center else 0)
            d.text((x+5,yy+6),l,font=tf,fill=(0,0,0,int(140*la))); d.text((x,yy),l,font=tf,fill=acc+(int(255*la),))
        if sub:
            yb=y0+len(lines)*lh+25; sa=ease((lt-0.35)/0.35); sf=F(B,50); w=d.textlength(sub,font=sf)
            x=(W-w)/2-35 if center else 70
            d.rounded_rectangle((x,yb,x+w+70,yb+92),46,fill=acc+(int(235*sa),)); d.text((x+35,yb+16),sub,font=sf,fill=(20,20,40,int(255*sa)))
    if kind=="list":
        la=ease(lt/0.4); ctext(d,"14 KURS",F(B,130),300+int(40*(1-la)),Y+(int(255*la),))
        ctext(d,"seni bekliyor",F(R,56),450,(255,255,255,int(255*la)))
        for n,c in enumerate(COURSES):
            col,row=n%2,n//2; ca=ease((lt-0.3-n*0.07)/0.3)
            x=70+col*480; y=620+row*118
            d.rounded_rectangle((x,y,x+460,y+96),26,fill=(255,255,255,int(38*ca)),outline=(255,255,255,int(110*ca)),width=2)
            f=F(B,38); 
            while d.textlength(c,font=f)>420: f=F(B,f.size-2)
            w=d.textlength(c,font=f); d.text((x+(460-w)/2,y+(96-f.size)/2-4),c,font=f,fill=(255,255,255,int(255*ca)))
    if kind=="cta":
        lines=["Hemen","kayıt ol!"]
        for n,l in enumerate(lines):
            la=ease((lt-n*0.12)/0.4); ctext(d,l,F(B,150),380+n*165+int(50*(1-la)),Y+(int(255*la),))
        tags=["9 yaş ve üzeri","En fazla 3 kurs seçebilirsin","Kontenjan sınırlı"]
        for n,tg in enumerate(tags):
            la=ease((lt-0.5-n*0.18)/0.35); f=F(B,46); w=d.textlength(tg,font=f); y=760+n*100
            d.rounded_rectangle(((W-w)/2-36,y,(W+w)/2+36,y+80),40,fill=(255,255,255,int(40*la)),outline=(255,255,255,int(150*la)),width=2)
            d.text(((W-w)/2,y+14),tg,font=f,fill=(255,255,255,int(255*la)))
        la=ease((lt-1.2)/0.4)
        ctext(d,"Kayıt linki açıklamada 👇".replace(" 👇",""),F(R,48),1110,(255,255,255,int(255*la)))
        ctext(d,"0258 241 07 46",F(R,52),1190,(255,255,255,int(255*la)))
        f=F(B,58); w=d.textlength("@gsb_denizligm",font=f); y=1290
        d.rounded_rectangle(((W-w)/2-50,y-10,(W+w)/2+50,y+90),50,fill=Y+(int(255*la),))
        d.text(((W-w)/2,y+8),"@gsb_denizligm",font=f,fill=(15,25,70,int(255*la)))
    d.rounded_rectangle((60,80,W-60,90),5,fill=(255,255,255,70))
    d.rounded_rectangle((60,80,60+int((W-120)*t/TOTAL),90),5,fill=(255,255,255,230))
    ft="GSB DENİZLİ GENÇLİK MERKEZİ"; w=d.textlength(ft,font=foot); d.text(((W-w)/2,H-230),ft,font=foot,fill=(255,255,255,200))
    out=Image.alpha_composite(fr,L).convert("RGB")
    if k>0 and lt<0.12: out=Image.blend(out,Image.new("RGB",(W,H),"white"),0.55*(1-lt/0.12))
    p.stdin.write(out.tobytes())
    if i==int(4.4*FPS): out.save(sys.argv[2],quality=92)
p.stdin.close(); p.wait()
print("duration",TOTAL)
