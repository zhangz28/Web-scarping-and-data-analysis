# Web-scraping-and-data-analysis

This project use the cumulative covid cases in orange county to inspect the spread 
of covid-19 in communities, especially in schools. The website COVID-19 Case Counts
and Testing Figures | Novel Coronavirus (COVID-19) (ochealthinfo.com) lists weekly
confirmed cases in schools and the data is updated weekly. 

1.Utilize selenium and Beautiful soup to scrape the confirmed cases in schools from 
Aug 16, 2020 to Apr 9, 2022 and savethem into a text file. 

2.Read the scraped data and do the following analyses and visualization:
      a.	Plot the confirmed cases over time. For each table, plot categories in each
          table in one figure, so that the numbers are comparable.
      b.  Use KNN smoothing method with Gaussian Kernel to predict how many people were
          infected during Thanksgiving and Christmas holiday two holidays.( In the Thanksgiving
          and Christmas holiday, the number of confirmed cases dropped down. The possible
          reason might be that people were traveling out of Orange County, even if they
          were infected, they did not take the test or took the covid test from anywhere else. )

