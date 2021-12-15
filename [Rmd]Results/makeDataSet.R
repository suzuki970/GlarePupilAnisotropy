
# params setting ------------------------------
sTime = -1
eTime = 4
timeLen = c(sTime,eTime)
nameOfVar = c("condition","condition")
f1 = "Locs"
f2 = "Pattern"

g1 <- rep(c("Upper","Lower","Center","Left","Right"),2)
g2 <- rep(c("Glare","Control"), times=c(5,5))

go1 <- c("Upper","Lower","Center","Left","Right")
go2 <- c("Glare","Control")

# micro-saccades ------------------------------
data=fromJSON(file="../[python]pre_processing/data/data20211124_f.json")

ind_data_ms = data.frame()
for(iTrial in 1:length(data[["sTimeOfMS"]])){
  numOfMS = length(data[["sTimeOfMS"]][[iTrial]])
  ind_data_ms = rbind(ind_data_ms,
                      data.frame(
                        sub = rep(data$sub[iTrial],numOfMS),
                        Locs = rep(g1[data$condition[iTrial]],numOfMS),
                        Pattern = rep(g2[data$condition[iTrial]],numOfMS),
                        data_y = data[["sTimeOfMS"]][[iTrial]]
                      ))
}

ind_data_ms[ind_data_ms$data_y < 0,]$data_y = 360+ind_data_ms[ind_data_ms$data_y < 0,]$data_y
ind_data_ms$DirCat = ind_data_ms$data_y

dat <- list((matrix(unlist(data$ampOfMS),nrow=length(data$ampOfMS),byrow=T)))
names(dat) <- c('y')
numOfTrial = dim(dat$y)[1]
numOfSub = length(unique(dat$sub))
lengthOfTime = dim(dat$y)[2]

x = seq(sTime,eTime,length=lengthOfTime)

ind_data_timeCourseMS <- data.frame(
  sub =  rep( data$sub, times = rep( lengthOfTime, numOfTrial)),
  data_y = t(matrix(t(dat$y),nrow=1)),
  data_x = x,
  Locs = rep(g1[data$condition], times = rep( lengthOfTime, numOfTrial)),
  Pattern = rep(g2[data$condition], times = rep( lengthOfTime, numOfTrial))
)

timeCourseMS_ave = aggregate( data_y ~ sub*data_x*Locs*Pattern, data = ind_data_timeCourseMS, FUN = "mean")

# gaze ------------------------------
gaze_data = data.frame(
  sub = data$sub,
  Locs = g1[data$condition],
  Pattern = g2[data$condition],
  gazeX = data$gazeX,
  gazeY = data$gazeY
)

gaze_data$numOfTrial = 0
for(iSub in unique(gaze_data$sub)){
  gaze_data[gaze_data$sub == iSub,]$numOfTrial = 1:dim(gaze_data[gaze_data$sub == iSub,])[1]
}

# pupil ------------------------------
dat <- list((matrix(unlist(data$PDR),nrow=length(data$PDR),byrow=T)),
            t(unlist(data$sub)),
            t(unlist(data$condition)))
names(dat) <- c('y', 'sub', 'condition')

ind_data = makePupilDataset_mat2long(dat,c('condition','condition'),timeLen,list(g1,g2),list(go1,go2),c(f1,f2))

data_e1 = aggregate( . ~ sub*data_x*Locs*Pattern, data = ind_data, FUN = "mean")
data_e1$minLatency = data$min

events_data = data.frame()
for(iSub in 1:length(data$events)){
  events_data = rbind(events_data,
                      data.frame(
    sub = unique(data_e1$sub)[iSub],
    events = unlist(data$events[iSub]),
    data_y = unlist(data$events_p[iSub]),
    tag = c('min',rep('ev',length(unlist(data$events[iSub]))-1))
  ))
}

# AUC ------------------------------
data_e1$Locs = factor(data_e1$Locs, go1)
numOfSub = length(unique(data_e1$sub))

data_auc = data.frame()
for(iSub in unique(data_e1$sub)){
  for(iLocs in unique(data_e1$Locs)){
    for(iPattern in unique(data_e1$Pattern)){
      tmp = data_e1[data_e1$sub == iSub &
                      data_e1$Locs == iLocs &
                      data_e1$Pattern == iPattern,]
      tmp = tmp[tmp$data_x > tmp$minLatency[1],]

      peakMin = data_e1[data_e1$sub == iSub &
                          data_e1$Locs == iLocs &
                          data_e1$Pattern == iPattern,]
      peakMin =  mean(peakMin[peakMin$data_x < tmp$minLatency[1]+0.25 &
                                peakMin$data_x > tmp$minLatency[1]-0.25,]$data_y)
      auc = 0
      for(ix in 1:(length(tmp$data_x))){
        auc = auc + ((tmp[ix,]$data_y-peakMin)*(1/500))
      }
      data_auc = rbind(data_auc,data.frame(
        sub = iSub,
        Locs = iLocs,
        Pattern = iPattern,
        data_y = c(peakMin,auc),
        comp = c('Early','Late')
      ))
    }
  }
}

# save data ------------------------------
save(data_e1,data_auc,
     events_data,gaze_data,
     ind_data_timeCourseMS,timeCourseMS_ave,ind_data_ms,
     file = "./data/dataset.rda")
