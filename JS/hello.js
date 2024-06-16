var temp = SELECT bt.saleareaid,
                o.fullname as saleareaid__orgname,
                (g.orgname ||'-' || pt.orgname ) as submitterid__orgname,
                bt.submitterid,p.userinfoid as submitterid__userinfoid,
                p.phonenumber as submitterid__userinfoid__phonenumber,
                store.storecode as storeid__storecode,
                store.storename as storeid__storename,
                store.storetype as storeid__storetype,
                d.dicvalue as storeid__storetype__dicvalue,
                store.storelevel as storeid__storelevel,
                dd.dicvalue as storeid__storelevel__dicvalue,
                ddd.dicvalue as storeid__channeltype__dicvalue,
                k.kasystemname as storeid__kaid__kasystemname,
                kkc.channelname as storeid__distributor__channelname,
                bt.submittime,
                bt.highcl ,
                bt.newcl,
                bt.boxcl,
                bt.indoor,
                bt.mpcl,
                bt.pricecl,
                bt.materialnum,
                bt.remark
            FROM kx_kq_biscuitca bt
            left join member g on g.orgstructid=bt.submitterid
            left join position pt on pt.orgstructid = g.parentorgstructid
            left join pl_userinfo p on p.userinfoid=g.userinfoid
            left join kx_kq_store store on store.id=bt.storeid
            left join pl_salearea o on o.orgstructid=store.seleareaid
            left join kx_kq_ka k on k.id=store.kaid
            left join kx_storetype d on d.dictionaryid=store.storetype
            left join kx_storelevel dd on dd.dictionaryid=store.storelevel
            left join kx_channelcustomertype ddd on ddd.dictionaryid=store.channeltype
            left join ka_kq_channelcustomers kkc on kkc.id = store.distributor
            WHERE store.storename is not null and bt.cltype = IN.kx_kq_biscuitca.cltype