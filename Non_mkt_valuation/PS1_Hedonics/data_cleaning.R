library(Hmisc)
library(dplyr)
library(data.table)
library(fastDummies)

        
setwd("~/Box/Non_mkt_valuation/PS1_Hedonics")
la_data1 <- read_csv("la_data1.csv", col_names = FALSE)
names(la_data1) <- c('house_ID', 'price', 'county', 'year_built', 'sq_feet', 'bath', 'bed', 'rooms', 'stories', 'violent_crime', 'property_crime', 'year_sale' )


LM <- price ~ bath bed stories property_crime property_crime^2 year_built year_built^2 sq_feet sq_feet^2 rooms rooms^2
la_data1$prop_crime_sq = la_data1$property_crime^2

linearMod <- lm(price ~ bath + bed + stories + property_crime + prop_crime_sq + poly(year_built,2, raw = TRUE) + poly(sq_feet, 2, raw = TRUE) + poly(rooms,2, raw = TRUE) + poly(violent_crime,2,raw = TRUE) + year_sale_1993 + year_sale_1994 + year_sale_1995+  year_sale_1996 + year_sale_1997 + year_sale_1998 + year_sale_2000 +  year_sale_2001 +  year_sale_2002 +  year_sale_2003 +  year_sale_2004 +  year_sale_2005 +  year_sale_2006 + year_sale_2007 + year_sale_2008+ county_59 + county_65 + county_71 + county_111, data=la_data1)  # build linear regression model on full data
print(linearMod)

linearMod <- lm(price ~ bath + bed, data=la_data1)  # build linear regression model on full data
print(linearMod)