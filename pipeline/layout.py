import numpy as np
import sys, cv2, math, os, sys


class Node:
    def __init__(self, label):
        self.label = label

        self.subs = None
        self.supers = None
        self.next = None
        
def angle_finder(x1,y1,x2,y2):
    myradians=math.atan2(y1-y2, x2-x1)
    return math.degrees(myradians)%360



def printTree(node):
    if node == None:
        return 0
    elif node.label!='\\frac':
        print(node.label, end="")
        if node.sub[0]!= None:
            print ('_{', end="")
            printTree(node.sub[0])
            print ('}', end="")
        if node.sup[0] != None:
            print ('^{', end="")
            printTree(node.sup[0])
            print ('}', end="")
        printTree(node.next[0])
    else:
        print(node.label, end="")
        print ('{', end="")
        if len(node.above)!=0:
            printTree(node.above[0])
        print ('}', end="")
        print ('{', end="")
        if len(node.below)!=0:
            printTree(node.below[0])
        print ('}', end="")
        printTree(node.next[0])

def get_latex(node):
    latex = ""
    if node == None:
        return ""
    elif node.label!='\\frac':
        latex = latex + node.label
        if node.sub[0]!= None:
            latex = latex + '_{'
            latex = latex + get_latex(node.sub[0])
            latex = latex + '}'
        if node.sup[0] != None:
            latex = latex + '^{' 
            latex = latex + get_latex(node.sup[0])
            latex = latex + '}'
        latex = latex + get_latex(node.next[0])
    else:
        latex = latex + node.label
        latex = latex + '{'
        if len(node.above)!=0:
            latex = latex + get_latex(node.above[0])
        latex = latex + '}'
        latex = latex +'{'
        if len(node.below)!=0:
             latex = latex + get_latex(node.below[0])
        latex = latex + '}'
       
        latex = latex + get_latex(node.next[0])

    return latex

def make_structure_tree(segment):
    print('SEGMENT ' , segment)
    if(segment == None):
        return Node('END')
    
    label = get_label(segment)
    node = Node(label)
    node.next = make_structure_tree(segment.next)
    node.supers = make_structure_tree(segment.sup)
    node.subs = make_structure_tree(segment.sub)

    return node
    
def check_equal(seg,segments):
    if seg!=0 and segments[seg-1].ar>4:
        if math.fabs(segments[seg-1].x-segments[seg].x)<15 and math.fabs(segments[seg-1].x+segments[seg-1].w-segments[seg].x-segments[seg].w)<15 :
            return seg-1
    if seg!=len(segments)-1 and segments[seg+1].ar>4:
        if math.fabs(segments[seg+1].x-segments[seg].x)<15 and math.fabs(segments[seg+1].x+segments[seg+1].w-segments[seg].x-segments[seg].w):
            return seg+1
    return -1   

def sanatize(segments,img):
    if len(segments)==0:
        return []
    segments_c = segments.copy()
    for seg in segments:
        try:
            seg_c=segments_c[segments_c.index(seg)]
        except:
            continue
        try:
            seg_c.ar=(seg_c.w/seg_c.h)
        except:
            seg_c.ar = 1

        if seg_c.ar>4:
            x=check_equal(segments.index(seg_c),segments)
            if x!=-1:
               
                if segments[x].y>seg_c.y:
                    seg_c.h=segments[x].y-seg_c.y+segments[x].h
                   
                elif segments[x].y<seg_c.y:
                    seg_c.h=seg_c.y-segments[x].y+seg_c.h
                    seg_c.y=segments[x].y
                    
                   
                # print("Seg h is: ",seg_c.h)
                seg_c.center=(seg_c.x+seg_c.w//2,seg_c.y+seg_c.h//2)
                segments_c.remove(segments[x])
                seg_c.label='='
                continue
            above=[]
            below=[]
            for sib in segments:
                if seg_c.x<sib.x and seg_c.x+seg_c.w>sib.x+sib.w:
                    if seg_c.center[1]<sib.center[1]:
                        # print("Below")
                        segments_c.remove(sib)
                        below.append(sib)
                    elif seg_c.center[1]>sib.center[1]:
                        # print("Above")
                        segments_c.remove(sib)
                        above.append(sib)

            

            if above!=[] or below!=[]:
                # print("Fraction")
                above=sanatize(above, img)
                above=label_neighbours(above)
                below=sanatize(below, img)
                below=label_neighbours(below)
                seg_c.above=above
                seg_c.below=below
                seg_c.label='\\frac'
                for segv in above:
                    if segv.next[0] != None:
                        nexts = segv.next[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),5)
                    if segv.sub[0] != None:
                        nexts = segv.sub[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),2)
                    if segv.sup[0] != None:
                        nexts = segv.sup[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),1)
                        
                for segv in below:
                    if segv.next[0] != None:
                        nexts = segv.next[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),5)
                        cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
                        cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
                    if segv.sub[0] != None:
                        nexts = segv.sub[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),2)
                        cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
                        cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
                    if segv.sup[0] != None:
                        nexts = segv.sup[0]
                        cv2.line(img,(segv.center[0],segv.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),1)
                        cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
                        cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
                        
            else:
                seg_c.label='-'

    return segments_c
def dist(seg1,seg2):
    return math.hypot(seg1.center[0]-seg2.center[0],seg1.center[1]-seg2.center[1])

def label_neighbours(segments):
    if len(segments)==0:
        return []
    segments[0].val=1
    # print("FINDING NEIGHBOURS")
    for seg in segments:
        for neigh in segments:
            angle = angle_finder(seg.center[0], seg.center[1], neigh.center[0], neigh.center[1])
            # print("angle:", angle)
            if((seg.center[0] ==  neigh.center[0]) and  (seg.center[1] == neigh.center[1]) ):
                # print("continue")
                continue
            # Removing back relation
            if(seg.center[0]-seg.w>neigh.center[0] or (angle>140 and angle<220)):
                continue
            if(math.fabs(seg.center[1]-neigh.center[1])<35):
                dis=neigh.x-seg.x-seg.w
                if(seg.val>=neigh.val and (neigh.back[0]== None or dist(seg,neigh)<dist(neigh,neigh.back[0]))):
                    if(seg.next[0] == None or dis < seg.next[1] ):
                        if seg.next[0]!=None:
                            seg.next[0].val=0
                        neigh.val=seg.val
                        seg.next = (neigh, dis)
                        if neigh.back[0]!=None:
                            neigh.back[0].next=(None,None)
                        if neigh.bsup[0]!=None:
                            neigh.bsup[0].sup=(None,None)
                        if neigh.bsub[0]!=None:
                            neigh.bsub[0].sub=(None,None)
                        neigh.back =(seg,dis)
            
            elif(angle >=15 and angle <= 110):
                dis  =  ( neigh.x ) 
                off = (neigh.h+seg.h)/2

                # print(1, dis)
                if seg.val>neigh.val:
                    if(neigh.bsup[0]== None or dist(seg,neigh)<dist(neigh,neigh.bsup[0])):
                        if(neigh.h<=seg.h+15 and seg.y- (neigh.y+neigh.h)<off ):
                            if(seg.sup[0] == None or dis < seg.sup[1]):
                                if seg.sup[0]!=None:
                                    seg.sup[0].val=0
                                neigh.val=seg.val/2
                                seg.sup = (neigh, dis)
                                if neigh.back[0]!=None:
                                    neigh.back[0].next=(None,None)
                                if neigh.bsup[0]!=None:
                                    neigh.bsup[0].sup=(None,None)
                                if neigh.bsub[0]!=None:
                                    neigh.bsub[0].sub=(None,None)
                                neigh.bsup =(seg,dis)
                                # print(seg.label+','+neigh.label)
            #sub s
            elif(angle >=250 and angle <= 330):
                dis  =  ( neigh.x )
                off = (neigh.h+seg.h)/2
                # print(1, dis)
                if seg.val>neigh.val:
                    if(neigh.bsub[0]== None or dist(seg,neigh)<dist(neigh,neigh.bsub[0])):
                        if(neigh.h<=seg.h+15 and (neigh.y)-(seg.y+seg.h)<off ):
                            if(seg.sub[0] == None or dis < seg.sub[1]):
                                if seg.sub[0]!=None:
                                    seg.sub[0].val=0
                                neigh.val=seg.val/2
                                seg.sub = (neigh, dis)
                                if neigh.back[0]!=None:
                                    neigh.back[0].next=(None,None)
                                if neigh.bsup[0]!=None:
                                    neigh.bsup[0].sup=(None,None)
                                if neigh.bsub[0]!=None:
                                    neigh.bsub[0].sub=(None,None)
                                neigh.bsub =(seg,dis)

        # if(seg.next[0]!=None and seg.sub[0]!=None and  dist(seg,seg.next[0])<dist(seg,seg.sub[0])):
        #     seg.sub=(None,None)
        # if(seg.next[0]!=None and seg.sup[0]!=None and  dist(seg,seg.next[0])<dist(seg,seg.sup[0])):
        #     seg.sup=(None,None)
        
            # if minus
    return segments

