library(corrplot)

FF <- read.csv("FF5.csv", header=TRUE)
excess_ret <- read.csv("excess_ret.csv", header=TRUE)
market_excess<- read.csv("market_excess.csv", header=TRUE)
df <- cbind(market_excess[2], FF[-1], excess_ret)
BAB <- read.csv("BAB.csv")
BAA <- read.csv("BAA.csv")
BAAB <- read.csv("BAAB.csv")

# Create empty dataframes
alpha_CAPM <- excess_ret[FALSE,]
beta_CAPM <- excess_ret[FALSE,]
alpha_FF5 <- excess_ret[FALSE,]
alpha_Car <- excess_ret[FALSE,]


# Regression
ptm <- proc.time()
total1 = ncol(df)
total2 = nrow(df)-11

for (n in c(10:total1)) {
# for (n in c(10:15)){
  for (i in c(1:total2)) {
  # for (i in c(1:(100-11))) {
    try({
      f_CAPM <- paste(colnames(df)[n], "~", paste("Market_Ret"))
      model_CAPM <- lm(f_CAPM, data = df[i:(i+11),])
      alpha_CAPM[i+11, c(eval(colnames(df)[n]))] <- model_CAPM$coefficients[1]
      beta_CAPM[i+11, c(eval(colnames(df)[n]))] <- model_CAPM$coefficients[2]
      
      f_FF5 <- paste(colnames(df)[n], "~", paste("Mkt.RF + SMB + HML + RMW + CMA"))
      model_FF5 <- lm(f_FF5, data = df[i:(i+11),])
      alpha_FF5[i+11, c(eval(colnames(df)[n]))] <- model_FF5$coefficients[1]
      
      f_Car <- paste(colnames(df)[n], "~", paste("Mkt.RF + SMB + HML + RMW + CMA + MOM"))
      model_Car <- lm(f_Car, data = df[i:(i+11),])
      alpha_Car[i+11, c(eval(colnames(df)[n]))] <- model_Car$coefficients[1]
      }, 
      silent=TRUE)
  }
}

proc.time() - ptm

# Add Date
alpha_CAPM[, c('Month')] <- excess_ret[, c('Month')]
beta_CAPM[, c('Month')] <- excess_ret[, c('Month')]
alpha_FF5[, c('Month')] <- excess_ret[, c('Month')]
alpha_Car[, c('Month')] <- excess_ret[, c('Month')]

# Write to csv files
write.csv(alpha_CAPM, 'alpha_CAPM.csv')
write.csv(beta_CAPM, 'beta_CAPM.csv')
write.csv(alpha_FF5, 'alpha_FF5.csv')
write.csv(alpha_Car, 'alpha_Car.csv')

# Correlation Matrix
corr <- cbind(BAA[2], BAB[2], BAAB[2], FF[-c(1,7)][61:576,])
corrmat <- cor(corr)
