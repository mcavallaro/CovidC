args <- commandArgs(trailingOnly = TRUE)
library(carData)
library(car)
library(questionr)

X_train = read.table(args[1], header=T, sep=',', row.names=1)
y_train = read.table(args[2], header=T, sep=',', row.names=1)
X_test = read.table(args[3], header=T, sep=',', row.names=1)

outcome <- names(y_train)
variables <- names(X_train)

f <- as.formula(
  paste(outcome,
        paste(variables, collapse = " + "), 
        sep = " ~ "))


logitMod <- glm(f, data=cbind(X_train, y_train), family=binomial(link="logit"))

oddsCI = odds.ratio(logitMod)
oddsCI2 = odds.ratio(logitMod, level=0.6826)
cat('alias:')
alias(logitMod)

VIF<-data.frame(vif(logitMod))
write.table(VIF, 'tmp.csv', sep=',')



predicted <- plogis(predict(logitMod, X_test))
              
#print(predict(logitMod, X_test)[1:5])


outfile = args[4]
pred_outfile = args[5]

summ = summary(logitMod)$coefficients

ci = confint.default(logitMod)

coefficients = merge(ci, summ, by=0)

# head(coefficients)
# head(as.data.frame(oddsCI))
# head(oddsCI)
coefficients = merge(coefficients, oddsCI,by.x='Row.names', by.y=0)
coefficients = merge(coefficients, oddsCI2,by.x='Row.names', by.y=0)


write.table(coefficients, outfile, sep=',')
write.table(predicted, pred_outfile, sep=',')
