import sys
sys.path = ['/nfs/slac/g/suncatfs/aidank/software/lib/python2.7/site-packages/matplotlib-1.5.1-py2.7-linux-x86_64.egg']+sys.path
import matplotlib.pyplot as plt
import csv
import numpy as np
import bisect

#reader=csv.reader(open('matches.csv','rb'))
#stuff = [line for line in reader][1:]

def get_champ_matrix(csv_name):
    reader=csv.reader(open(csv_name,'rb'))
    stuff = [line for line in reader][1:]
    champ_matrix = np.zeros((len(stuff)*2,0))
    champions = []
    wins = []
    champ_inds1 = [25,28,31,34,37]
    champ_inds2= [41,44,47,50,53]
    for i,line in enumerate(stuff):
        wins.append(int(line[5]))
        wins.append(int(not int(line[5])))
        for cind in champ_inds1:
            try:
                ind = champions.index(line[cind].lower())
                champ_matrix[2*i][ind]=1 
            except ValueError:
                champions.append(line[cind].lower())
                champ_matrix=np.hstack((champ_matrix,np.zeros((champ_matrix.shape[0],1))))
                champ_matrix[2*i][-1]=1
        for cind in champ_inds2:
            try:
                ind = champions.index(line[cind].lower())
                champ_matrix[2*i+1][ind]=1
            except ValueError:
                champions.append(line[cind].lower())
                champ_matrix=np.hstack((champ_matrix,np.zeros((champ_matrix.shape[0],1))))
                champ_matrix[2*i+1][-1]=1

    return np.array(wins),champ_matrix,champions

def getBasicStats(csv_name,write=None):
    reader=csv.reader(open(csv_name,'rb'))
    stuff = [line for line in reader][1:]
    headings = ['Match','Result','totalGold','goldDiff','totalKills','totalDeaths','totalAssists','KD_rat','indGold','indKills','indDeaths','indAssists']
    stats=[]

    for i,line in enumerate(stuff):
        bStats=[]
        rstats=[]
        bRes = eval(line[5])
        rRes= int(not bRes)
        bGold = eval(line[10])[-1]
        rGold = eval(line[17])[-1]
        bGD = eval(line[9])[-1]
        rGD = -1*bGD
        big = [eval(ig)[-1] for ig in np.array(line)[[26,29,32,35,38]]]
        rig = [eval(ig)[-1] for ig in np.array(line)[[42,45,48,51,54]]]
        bRoles = list(np.array(line)[[24,27,30,33,36]])
        rRoles = list(np.array(line)[[40,43,46,49,52]])
        bkills = eval(line[11])
        rkills = eval(line[18])
        try:
            bk,rk,bd,rd,ba,ra,bik,rik,bid,rid,bia,ria=getKillData(bkills,rkills,bRoles,rRoles)        
        except ValueError:
            raise ValueError('%i' % i)
        bStats = ['B%i' % i, bRes,bGold,bGD,bk,bd,ba,bk-bd,big,bik,bid,bia]
        rStats = ['R%i' % i, rRes,rGold,rGD,rk,rd,ra,rk-rd,rig,rik,rid,ria]
        stats.append(bStats)
        stats.append(rStats)
        if write:
            writer = csv.writer(open(write,'wb')).writerows([headings]+stats)

    return stats


def getKillData(bKills,rKills,bRoles,rRoles):
    bk,rk,bd,rd,ba,ra=[0,0,0,0,0,0]
    bik = [0,0,0,0,0]
    rik = [0,0,0,0,0]
    bid = [0,0,0,0,0]
    rid = [0,0,0,0,0]
    bia = [0,0,0,0,0]
    ria = [0,0,0,0,0]
    for kill in bKills:
        if kill[1]=='TooEarly':
            continue
        bk+=1
        rd+=1
        ba+=len(kill[3])
        try:
            k_ind = bRoles.index(' '.join(kill[2].split(' ')[1:]).strip())
        except ValueError:
            k_ind = bRoles.index(kill[2].strip())
        try:
            v_ind = rRoles.index(' '.join(kill[1].split(' ')[1:]).strip())
        except ValueError:
            v_ind=rRoles.index(kill[1].strip())
        a_inds =[]
        for assist in kill[3]:
            try:    
                a_inds.append(bRoles.index(' '.join(assist.split(' ')[1:]).strip()))
            except ValueError:
                a_inds.append(bRoles.index(assist.strip()))
        bik[k_ind]+=1
        rid[v_ind]+=1
        for aind in a_inds:
            bia[aind]+=1 
    for kill in rKills:
        if kill[1]=='TooEarly':
            continue
        rk+=1
        bd+=1
        ra+=len(kill[3])
        try:
            k_ind = rRoles.index(' '.join(kill[2].split(' ')[1:]).strip())
        except ValueError:
            k_ind = rRoles.index(kill[2].strip())
        try:
            v_ind = bRoles.index(' '.join(kill[1].split(' ')[1:]).strip())
        except ValueError:
            v_ind=bRoles.index(kill[1].strip())
        a_inds =[]
        for assist in kill[3]:
            try:
                a_inds.append(rRoles.index(' '.join(assist.split(' ')[1:]).strip()))
            except ValueError:
                a_inds.append(rRoles.index(assist.strip()))
        rik[k_ind]+=1
        bid[v_ind]+=1
        for aind in a_inds:
            ria[aind]+=1
    return bk,rk,bd,rd,ba,ra,bik,rik,bid,rid,bia,ria


def get_hist_bins(stats,gold_bins=None,gd_bins=None,kill_bins=None,death_bins=None,assist_bins=None,kd_bins=None):
    #stats = stats[:8]
    stats = np.array(stats[1:],dtype='object')
    gold = stats[:,2]
    GD=stats[:,3]
    kills = stats[:,4]
    deaths = stats[:,5]
    assists = stats[:,6]
    kds = stats[:,7]
    gold_r = (np.min(gold),np.max(gold))
    GD_r = (np.min(GD),np.max(GD))
    kills_r=(np.min(kills),np.max(kills))
    deaths_r = (np.min(deaths),np.max(deaths))
    assists_r = (np.min(assists),np.max(assists))
    kds_r = (np.min(kds),np.max(kds))
    hists=[]
    ranges = [gold_r,GD_r,kills_r,deaths_r,assists_r,kds_r]
    set_bins = [gold_bins,gd_bins,kill_bins,death_bins,assist_bins,kd_bins]
    for k,r in enumerate(ranges):
        print set_bins[k]
        if set_bins[k]:
            bins=set_bins[k]
        else:
            bins = np.linspace(r[0],r[1],9,endpoint=False)
        hists.append([bins,[]])
        for b in range(len(bins)):
            hists[k][1].append([])
    #print len(hists),len(hists[0]),len(hists[0][0]), len(hists[0][1]),len(hists[0][1][0])
    #print hists[0][1]
    for i,row in enumerate(stats):
        #print row
        for j in range(6):
            #print row[2+j]
            #print hists[i][0]
            ind = np.digitize([row[2+j]],hists[j][0])[0]-1
            #print ind,row[2+j],hists[j][0]
            hists[j][1][ind].append(row[1])
    return hists

def make_win_bar_plots(hists):
    x_labels=['Gold','Gold Difference','Kills','Deaths','Assists','Kills-Deaths']
    for i,hist in enumerate(hists):
        bins=hist[0]
        values = [np.sum(hist[1][j])/float(len(hist[1][j])) for j in range(len(bins))]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(values))+1,values,align='edge',color='b',width=1)
        ax.set_xticks(np.arange(len(values))+1)
        ax.set_xticklabels(bins)
        ax.set_ylabel('Win Percentage')
        ax.set_xlabel(x_labels[i])
        fig.savefig('%s vs Win Percentage.pdf' % x_labels[i])
   
def get_role_stats(stats):
    #stats = stats[1:]
    mstats = np.array(stats,dtype='object')[:,8:]
    bstats = np.empty((4,mstats.shape[0],5))
    for i,ms in enumerate(mstats):
        for k in range(4):
            bstats[k][i]=ms[k]
    rstats=[]
    for i in range(5):
        temp = bstats[:,:,i]
        rstats.append([get_stats(temp[0]),get_stats(temp[1]),get_stats(temp[2]),get_stats(temp[3])])
    return bstats,rstats
            
def get_stats(ind_stats):
    mean = np.mean(ind_stats)
    median =np.median(ind_stats)
    mini = np.min(ind_stats)
    maxu = np.max(ind_stats)
    return [mean,median,mini,maxu]

def make_role_histograms(role_data):
    roles = ['Top','Jungler','Middle','AD Carry','Support']
    colors=['gold','g','mediumblue','r','orchid']
    #Make a 2x2 plot for each role
    for i,role in enumerate(roles):
        fig,axes = plt.subplots(2,2)
        print role_data[0,:,i].shape
        axes[0,0].hist(role_data[0,:,i],bins='auto',facecolor=colors[i])
        axes[0,1].hist(role_data[2,:,i],bins='auto',facecolor=colors[i])
        axes[1,0].hist(role_data[1,:,i],bins='auto',facecolor=colors[i])
        axes[1,1].hist(role_data[3,:,i],bins='auto',facecolor=colors[i])
        axes[0,0].tick_params(axis='both',which='major',labelsize=8.5)
        axes[1,0].tick_params(axis='both',which='major',labelsize=8.5)
        axes[0,1].tick_params(axis='both',which='major',labelsize=8.5)
        axes[1,1].tick_params(axis='both',which='major',labelsize=8.5)
        axes[0,0].set_ylabel('Count',size=8.5)
        axes[0,1].set_ylabel('Count',size=8.5)
        axes[1,0].set_ylabel('Count',size=8.5)
        axes[1,1].set_ylabel('Count',size=8.5)
        axes[0,0].set_xlabel('Gold',size=8.5)
        axes[0,1].set_xlabel('Deaths',size=8.5)
        axes[1,0].set_xlabel('Kills',size=8.5)
        axes[1,1].set_xlabel('Assists',size=8.5)
        #axes[0,0].set_title('Gold')
        #axes[0,1].set_title('Deaths')
        #axes[1,0].set_title('Kills')
        #axes[1,1].set_title('Assists')
        plt.suptitle('%s Stat Graph' % roles[i])
        fig.savefig('%s_stat_graph.pdf' % roles[i])

